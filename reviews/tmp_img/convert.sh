#!/bin/bash
images=$(find *.svg)

for img in $(find *.svg)
do
	out=${img/svg/eps}
	inkscape $img -E $out --export-ignore-filters --export-ps-level=3
done
rm *.svg