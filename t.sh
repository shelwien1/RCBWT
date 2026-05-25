#!/bin/sh
set -e
./rcbwt book1 book1out book1bwt
cmp book1out book1bwt
