#!/usr/bin/env bash
site_name=aa
sp_end_point=localhost:7002:7003
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/../transfer
sleep 1
$DIR/sub_start.sh $site_name $sp_end_point &
