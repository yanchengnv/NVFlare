#!/usr/bin/env bash
site_name=ba
sp_end_point=localhost:9002:9003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/../transfer
sleep 1
$DIR/sub_start.sh $site_name $sp_end_point &
