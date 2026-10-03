#!/usr/bin/env bash
# Keeps routing to the arm correct whether it's wired through the switch
# (enp55s0) or bridged onto WiFi (SmartFactory/wlo1). A /32 host route
# always wins over the /24 subnet route regardless of metric, so whichever
# link the arm is NOT on must have its override route removed, and
# whichever link it IS on (when it's WiFi-only) must have one added.
set -euo pipefail

ARM_IP="192.168.1.232"
WIRED_IF="enp55s0"
WIFI_IF="wlo1"
WIFI_CONN="SmartFactory"

wired_ok=false
wifi_ok=false
ping -c1 -W1 -I "$WIRED_IF" "$ARM_IP" >/dev/null 2>&1 && wired_ok=true
ping -c1 -W1 -I "$WIFI_IF" "$ARM_IP" >/dev/null 2>&1 && wifi_ok=true

current_route="$(nmcli -g ipv4.routes connection show "$WIFI_CONN" 2>/dev/null || true)"
has_override=false
[[ "$current_route" == *"${ARM_IP}/32"* ]] && has_override=true

if $wired_ok; then
    if $has_override; then
        nmcli connection modify "$WIFI_CONN" ipv4.routes ""
        nmcli device reapply "$WIFI_IF" >/dev/null
        logger -t arm-route-autofix "arm reachable via $WIRED_IF; removed WiFi override route"
    fi
elif $wifi_ok; then
    if ! $has_override; then
        nmcli connection modify "$WIFI_CONN" +ipv4.routes "${ARM_IP}/32"
        nmcli device reapply "$WIFI_IF" >/dev/null
        logger -t arm-route-autofix "arm reachable via $WIFI_IF only; added WiFi override route"
    fi
else
    logger -t arm-route-autofix "arm ($ARM_IP) not reachable on $WIRED_IF or $WIFI_IF"
fi
