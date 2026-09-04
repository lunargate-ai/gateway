package config

import "testing"

func TestServerAddressUsesHostPortEncoding(t *testing.T) {
	tests := []struct {
		name string
		host string
		want string
	}{
		{name: "empty host", want: ":8080"},
		{name: "IPv4 wildcard", host: "0.0.0.0", want: "0.0.0.0:8080"},
		{name: "IPv4 loopback", host: "127.0.0.1", want: "127.0.0.1:8080"},
		{name: "IPv6 wildcard", host: "::", want: "[::]:8080"},
		{name: "IPv6 loopback", host: "::1", want: "[::1]:8080"},
		{name: "bracketed IPv6", host: "[::1]", want: "[::1]:8080"},
		{name: "zoned IPv6", host: "[fe80::1%lo0]", want: "[fe80::1%lo0]:8080"},
		{name: "hostname", host: "gateway.internal", want: "gateway.internal:8080"},
		{name: "absolute hostname", host: "gateway.internal.", want: "gateway.internal.:8080"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := ServerConfig{Host: test.host, Port: 8080}
			if got := cfg.Address(); got != test.want {
				t.Fatalf("Address() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestValidateServerConfigRejectsMalformedHost(t *testing.T) {
	tests := []struct {
		name    string
		host    string
		wantErr bool
	}{
		{name: "empty wildcard"},
		{name: "IPv4 wildcard", host: "0.0.0.0"},
		{name: "IPv6 wildcard", host: "::"},
		{name: "IPv4", host: "10.2.0.153"},
		{name: "IPv6", host: "2001:db8::1"},
		{name: "bracketed IPv6", host: "[2001:db8::1]"},
		{name: "hostname", host: "gateway.internal"},
		{name: "localhost", host: "localhost"},
		{name: "empty brackets", host: "[]", wantErr: true},
		{name: "missing closing bracket", host: "[::1", wantErr: true},
		{name: "missing opening bracket", host: "::1]", wantErr: true},
		{name: "nested brackets", host: "[[::1]]", wantErr: true},
		{name: "extra closing bracket", host: "[::1]]", wantErr: true},
		{name: "bracketed IPv4", host: "[127.0.0.1]", wantErr: true},
		{name: "bracketed hostname", host: "[gateway.internal]", wantErr: true},
		{name: "host and port", host: "gateway.internal:8080", wantErr: true},
		{name: "space", host: "bad host", wantErr: true},
		{name: "slash", host: "bad/host", wantErr: true},
		{name: "empty label", host: "bad..host", wantErr: true},
		{name: "leading hyphen", host: "-bad.example", wantErr: true},
		{name: "trailing hyphen", host: "bad-.example", wantErr: true},
		{name: "invalid IPv4", host: "999.1.1.1", wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := ServerConfig{Host: test.host, Port: 8080}
			err := validateServerConfig(cfg)
			if test.wantErr && err == nil {
				t.Fatalf("validateServerConfig(%q) returned nil error", test.host)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("validateServerConfig(%q) returned error: %v", test.host, err)
			}
			if test.wantErr && cfg.Address() == ":8080" {
				t.Fatalf("Address(%q) degraded to a wildcard listen address", test.host)
			}
		})
	}
}
