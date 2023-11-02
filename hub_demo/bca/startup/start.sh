#!/usr/bin/env bash
site_name=bca
sp_end_point=localhost:6002:6003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/../transfer
sleep 1
$DIR/sub_start.sh $site_name $sp_end_point &
