#!/usr/bin/env bash
# Record the new version (X.Y.Z) in CITATION.cff. Run by release.yml in the
# release commit; the package version itself comes from the tag (hatch-vcs).
set -euo pipefail
version="${1:?usage: set_version.sh X.Y.Z}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not a version: $version" >&2; exit 1; }
perl -pi -e "s/^version: .*/version: $version/; s/^date-released: .*/date-released: $(date -u +%Y-%m-%d)/" CITATION.cff
