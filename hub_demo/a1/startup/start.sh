#!/usr/bin/env bash
site_name=a1
sp_end_point=localhost:8002:8003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/../transfer
sleep 1
$DIR/sub_start.sh $site_name $sp_end_point &
