#!/bin/bash
./imagelist_creator ./imagelist.xml ./test/*.png
./a.out -m=generic -w=7 -h=10 -a=1.0 -sw=80.0 -sh=80.0 -o=camera.xml -su -oe ./imagelist.xml
