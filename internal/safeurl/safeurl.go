// Package safeurl builds HTTP endpoints and removes credential-bearing URL
// components from errors before they cross an observability boundary.
package safeurl

import (
	"errors"
	"net/url"
	"sort"
	"strings"
)

var ErrInvalidHTTPBaseURL = errors.New("invalid absolute HTTP base URL")

// ParseHTTPBaseURL parses an absolute HTTP(S) base URL without echoing the
// supplied value in validation errors.
func ParseHTTPBaseURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.Opaque != "" || parsed.Hostname() == "" {
		return nil, ErrInvalidHTTPBaseURL
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, ErrInvalidHTTPBaseURL
	}
	return parsed, nil
}

// JoinHTTPPath appends escaped path elements while preserving any base query.
// URL fragments never participate in HTTP requests and are intentionally
// discarded. Userinfo remains available to net/http but must be redacted from
// errors with RedactTransportError.
func JoinHTTPPath(rawBase string, elem ...string) (string, error) {
	joined, err := joinHTTPPath(rawBase, elem...)
	if err != nil {
		return "", err
	}
	return joined.String(), nil
}

// JoinHTTPPathAndRawQuery preserves the base query and appends the request
// query after it. This keeps configured query authentication intact for native
// passthrough operations without re-encoding extension parameters.
func JoinHTTPPathAndRawQuery(rawBase string, rawQuery string, elem ...string) (string, error) {
	joined, err := joinHTTPPath(rawBase, elem...)
	if err != nil {
		return "", err
	}
	if rawQuery != "" {
		if joined.RawQuery == "" {
			joined.RawQuery = rawQuery
		} else {
			joined.RawQuery += "&" + rawQuery
		}
	}
	return joined.String(), nil
}

func joinHTTPPath(rawBase string, elem ...string) (*url.URL, error) {
	base, err := ParseHTTPBaseURL(rawBase)
	if err != nil {
		return nil, err
	}
	joined := base.JoinPath(elem...)
	joined.Fragment = ""
	joined.RawFragment = ""
	return joined, nil
}

// RedactedHTTPURL returns an HTTP(S) URL without userinfo, query, or fragment.
func RedactedHTTPURL(raw string) (string, bool) {
	parsed, err := ParseHTTPBaseURL(raw)
	if err != nil {
		return "", false
	}
	redactURL(parsed)
	return parsed.String(), true
}

// RedactTransportError clones a url.Error so callers retain errors.Is,
// errors.As, Timeout, Temporary, and the transport category while its text no
// longer exposes userinfo, query parameters, or fragments. requestURL should
// be the URL passed to http.Client.Do.
func RedactTransportError(err error, requestURL *url.URL) error {
	if err == nil {
		return nil
	}
	urlErr, ok := err.(*url.Error)
	if !ok {
		return err
	}

	redacted := *urlErr
	redactedURL, replacements := redactedURLAndReplacements(urlErr.URL, requestURL)
	redacted.URL = redactedURL
	if urlErr.Err != nil {
		redacted.Err = &redactedCause{
			cause:   urlErr.Err,
			message: replaceSensitiveURLText(urlErr.Err.Error(), replacements),
		}
	}
	return &redacted
}

type redactedCause struct {
	cause   error
	message string
}

func (e *redactedCause) Error() string { return e.message }
func (e *redactedCause) Unwrap() error { return e.cause }

func (e *redactedCause) Timeout() bool {
	timeout, ok := e.cause.(interface{ Timeout() bool })
	return ok && timeout.Timeout()
}

func (e *redactedCause) Temporary() bool {
	temporary, ok := e.cause.(interface{ Temporary() bool })
	return ok && temporary.Temporary()
}

type textReplacement struct {
	old string
	new string
}

func redactedURLAndReplacements(raw string, requestURL *url.URL) (string, []textReplacement) {
	replacements := make([]textReplacement, 0, 8)
	redacted := "[redacted-url]"
	if parsed, err := url.Parse(raw); err == nil && parsed != nil {
		redactedCopy := *parsed
		redactURL(&redactedCopy)
		if value := redactedCopy.String(); value != "" {
			redacted = value
		}
		replacements = appendURLReplacements(replacements, parsed, redacted)
	}
	if requestURL != nil {
		requestCopy := *requestURL
		redactURL(&requestCopy)
		if value := requestCopy.String(); value != "" {
			redacted = value
		}
		replacements = appendURLReplacements(replacements, requestURL, redacted)
	}
	sort.SliceStable(replacements, func(i, j int) bool {
		return len(replacements[i].old) > len(replacements[j].old)
	})
	return redacted, replacements
}

func appendURLReplacements(replacements []textReplacement, parsed *url.URL, redacted string) []textReplacement {
	if parsed == nil {
		return replacements
	}
	if raw := parsed.String(); raw != "" {
		replacements = append(replacements, textReplacement{old: raw, new: redacted})
	}
	if parsed.User != nil {
		if userinfo := parsed.User.String(); userinfo != "" {
			replacements = append(replacements, textReplacement{old: userinfo, new: "[redacted-userinfo]"})
		}
		if username := parsed.User.Username(); username != "" {
			replacements = append(replacements, textReplacement{old: username, new: "[redacted-user]"})
		}
		if password, ok := parsed.User.Password(); ok && password != "" {
			replacements = append(replacements, textReplacement{old: password, new: "[redacted-password]"})
		}
	}
	if parsed.RawQuery != "" {
		replacements = append(replacements, textReplacement{old: parsed.RawQuery, new: "[redacted-query]"})
	}
	if parsed.Fragment != "" {
		replacements = append(replacements, textReplacement{old: parsed.Fragment, new: "[redacted-fragment]"})
	}
	return replacements
}

func replaceSensitiveURLText(message string, replacements []textReplacement) string {
	for _, replacement := range replacements {
		if replacement.old != "" {
			message = strings.ReplaceAll(message, replacement.old, replacement.new)
		}
	}
	return message
}

func redactURL(parsed *url.URL) {
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	parsed.RawFragment = ""
}
