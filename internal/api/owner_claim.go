package api

type ownerClaimResult uint8

const (
	ownerClaimUnavailable ownerClaimResult = iota
	ownerClaimed
	ownerClaimRefreshed
	ownerClaimConflict
)

func (r ownerClaimResult) retained() bool {
	return r == ownerClaimed || r == ownerClaimRefreshed
}

type ownerLookupResult uint8

const (
	ownerLookupMissing ownerLookupResult = iota
	ownerLookupBound
	ownerLookupConflict
)
