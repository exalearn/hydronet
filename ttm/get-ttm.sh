#! /bin/bash

wget -c https://sites.uw.edu/wdbase/files/2019/01/pot_ttm-1p9hi7d.zip
unzip -o pot_ttm-1p9hi7d.zip
cd pot_ttm
make --file ../Makefile.ttm clean all
