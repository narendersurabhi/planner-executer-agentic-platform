#!/usr/bin/env bash

set -euo pipefail

apply_changes=0
assume_yes=0
include_xcode=0
include_brew=0
include_docker=0
include_trash=0
include_time_machine=0

log_days="${LOG_DAYS:-14}"
xcode_archive_days="${XCODE_ARCHIVE_DAYS:-30}"
total_known_kb=0

usage() {
  cat <<'EOF'
Usage: scripts/cleanup_macos_system_data.sh [options]

Targets common sources of macOS "System Data" bloat. The default mode is a dry run.

Safe default targets:
  - ~/Library/Caches
  - files in ~/Library/Logs older than LOG_DAYS
  - files in ~/Library/Logs/DiagnosticReports older than LOG_DAYS

Optional targets:
  --xcode          Clear Xcode DerivedData, simulator caches, iOS DeviceSupport,
                   and archives older than XCODE_ARCHIVE_DAYS
  --brew           Run "brew cleanup -s"
  --docker         Run "docker system prune -af --volumes"
  --trash          Empty ~/.Trash
  --time-machine   Delete local Time Machine snapshots (best run with sudo)
  --all-extra      Enable all optional targets

Control flags:
  --apply          Delete data instead of only reporting it
  -y, --yes        Skip the confirmation prompt when used with --apply
  -h, --help       Show this help text

Environment variables:
  LOG_DAYS             Default: 14
  XCODE_ARCHIVE_DAYS   Default: 30

Examples:
  scripts/cleanup_macos_system_data.sh
  scripts/cleanup_macos_system_data.sh --xcode --trash
  scripts/cleanup_macos_system_data.sh --apply --xcode --brew --trash
  sudo scripts/cleanup_macos_system_data.sh --apply --time-machine
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

human_kb() {
  awk -v kb="${1:-0}" 'BEGIN {
    split("KB MB GB TB", units, " ");
    value = kb + 0;
    idx = 1;
    while (value >= 1024 && idx < 4) {
      value /= 1024;
      idx++;
    }
    printf("%.1f %s", value, units[idx]);
  }'
}

path_kb() {
  local path="$1"

  if [[ ! -e "$path" ]]; then
    echo 0
    return
  fi

  du -sk "$path" 2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}

find_kb() {
  local root="$1"
  shift

  if [[ ! -e "$root" ]]; then
    echo 0
    return
  fi

  find "$root" "$@" -exec du -sk {} + 2>/dev/null | awk '{sum += $1} END {print sum + 0}'
}

note_target() {
  local label="$1"
  local kb="${2:-0}"

  total_known_kb=$((total_known_kb + kb))
  printf "  %-36s %10s\n" "$label" "$(human_kb "$kb")"
}

note_unknown_target() {
  local label="$1"
  printf "  %-36s %10s\n" "$label" "unknown"
}

clear_dir_contents() {
  local path="$1"

  if [[ -d "$path" ]]; then
    find "$path" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null || true
  fi
}

delete_old_files() {
  local path="$1"
  local days="$2"

  if [[ -d "$path" ]]; then
    find "$path" -type f -mtime +"$days" -exec rm -f -- {} + 2>/dev/null || true
    find "$path" -type d -empty -delete 2>/dev/null || true
  fi
}

clean_user_caches() {
  local path="$HOME/Library/Caches"
  note_target "User caches" "$(path_kb "$path")"

  if [[ "$apply_changes" -eq 1 ]]; then
    clear_dir_contents "$path"
  fi
}

clean_logs() {
  local path="$HOME/Library/Logs"
  local kb

  kb="$(find_kb "$path" -type f -mtime +"$log_days")"
  note_target "Logs older than ${log_days} days" "$kb"

  if [[ "$apply_changes" -eq 1 ]]; then
    delete_old_files "$path" "$log_days"
  fi
}

clean_diagnostic_reports() {
  local path="$HOME/Library/Logs/DiagnosticReports"
  local kb

  kb="$(find_kb "$path" -type f -mtime +"$log_days")"
  note_target "Diagnostic reports > ${log_days} days" "$kb"

  if [[ "$apply_changes" -eq 1 ]]; then
    delete_old_files "$path" "$log_days"
  fi
}

clean_xcode() {
  local derived_data="$HOME/Library/Developer/Xcode/DerivedData"
  local device_support="$HOME/Library/Developer/Xcode/iOS DeviceSupport"
  local simulator_caches="$HOME/Library/Developer/CoreSimulator/Caches"
  local archives="$HOME/Library/Developer/Xcode/Archives"
  local total_kb=0
  local derived_kb=0
  local device_support_kb=0
  local simulator_cache_kb=0
  local archives_kb=0

  derived_kb="$(path_kb "$derived_data")"
  device_support_kb="$(path_kb "$device_support")"
  simulator_cache_kb="$(path_kb "$simulator_caches")"
  archives_kb="$(find_kb "$archives" -type d -name '*.xcarchive' -mtime +"$xcode_archive_days")"
  total_kb=$((derived_kb + device_support_kb + simulator_cache_kb + archives_kb))

  note_target "Xcode cleanup set" "$total_kb"

  if [[ "$apply_changes" -ne 1 ]]; then
    return
  fi

  clear_dir_contents "$derived_data"
  clear_dir_contents "$device_support"
  clear_dir_contents "$simulator_caches"

  if [[ -d "$archives" ]]; then
    find "$archives" -type d -name '*.xcarchive' -mtime +"$xcode_archive_days" -exec rm -rf -- {} + 2>/dev/null || true
    find "$archives" -type d -empty -delete 2>/dev/null || true
  fi

  if has_cmd xcrun; then
    xcrun simctl delete unavailable >/dev/null 2>&1 || true
  fi
}

clean_homebrew() {
  local brew_cache=""
  local kb=0

  if ! has_cmd brew; then
    return
  fi

  brew_cache="$(brew --cache 2>/dev/null || true)"
  if [[ -n "$brew_cache" ]]; then
    kb="$(path_kb "$brew_cache")"
  fi

  note_target "Homebrew cache cleanup" "$kb"

  if [[ "$apply_changes" -eq 1 ]]; then
    brew cleanup -s >/dev/null
  fi
}

clean_docker() {
  if ! has_cmd docker; then
    return
  fi

  note_unknown_target "Docker system prune"

  if [[ "$apply_changes" -eq 1 ]]; then
    docker system prune -af --volumes >/dev/null
  fi
}

clean_trash() {
  local path="$HOME/.Trash"
  note_target "Trash" "$(path_kb "$path")"

  if [[ "$apply_changes" -eq 1 ]]; then
    clear_dir_contents "$path"
  fi
}

list_time_machine_snapshots() {
  if ! has_cmd tmutil; then
    return
  fi

  tmutil listlocalsnapshots / 2>/dev/null | awk -F. '/com\.apple\.TimeMachine\./ {print $(NF-1)}'
}

clean_time_machine() {
  local snapshots
  local count

  if ! has_cmd tmutil; then
    return
  fi

  snapshots="$(list_time_machine_snapshots || true)"
  count="$(printf "%s\n" "$snapshots" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [[ "$count" -eq 0 ]]; then
    return
  fi

  printf "  %-36s %10s\n" "Time Machine local snapshots" "${count} items"

  if [[ "$apply_changes" -ne 1 ]]; then
    return
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Skipping Time Machine snapshots: re-run with sudo to delete them." >&2
    return
  fi

  while IFS= read -r snapshot; do
    [[ -z "$snapshot" ]] && continue
    tmutil deletelocalsnapshots "$snapshot" >/dev/null 2>&1 || true
  done <<< "$snapshots"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      apply_changes=1
      shift
      ;;
    -y|--yes)
      assume_yes=1
      shift
      ;;
    --xcode)
      include_xcode=1
      shift
      ;;
    --brew)
      include_brew=1
      shift
      ;;
    --docker)
      include_docker=1
      shift
      ;;
    --trash)
      include_trash=1
      shift
      ;;
    --time-machine)
      include_time_machine=1
      shift
      ;;
    --all-extra)
      include_xcode=1
      include_brew=1
      include_docker=1
      include_trash=1
      include_time_machine=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "macOS System Data cleanup"
echo
echo "Planned targets:"
clean_user_caches
clean_logs
clean_diagnostic_reports

if [[ "$include_xcode" -eq 1 ]]; then
  clean_xcode
fi

if [[ "$include_brew" -eq 1 ]]; then
  clean_homebrew
fi

if [[ "$include_docker" -eq 1 ]]; then
  clean_docker
fi

if [[ "$include_trash" -eq 1 ]]; then
  clean_trash
fi

if [[ "$include_time_machine" -eq 1 ]]; then
  clean_time_machine
fi

echo
echo "Known reclaimable space: $(human_kb "$total_known_kb")"
echo

if [[ "$apply_changes" -eq 0 ]]; then
  cat <<'EOF'
Dry run only. No files were deleted.

Re-run with --apply to perform the cleanup.
Examples:
  scripts/cleanup_macos_system_data.sh --apply
  scripts/cleanup_macos_system_data.sh --apply --xcode --trash
  sudo scripts/cleanup_macos_system_data.sh --apply --time-machine
EOF
  exit 0
fi

if [[ "$assume_yes" -ne 1 ]]; then
  printf "Apply this cleanup now? [y/N] "
  read -r response
  case "$response" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Aborted."
      exit 1
      ;;
  esac
fi

echo
echo "Cleanup completed."
