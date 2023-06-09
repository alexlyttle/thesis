#!/bin/bash
## Summarise the thesis PDF
file='thesis.pdf'
count=''
if type pdfinfo >/dev/null 2>/dev/null; then
    count=`pdfinfo -- $file | sed -n '/^Pages:/ s/.*[^0-9]//p'`
    echo "Page count: $count pages"
else
  echo "Page count unavailable - requires tool 'pdfinfo'."
fi

if type gs >/dev/null 2>/dev/null; then
    bw=`gs -q -o - -sDEVICE=inkcov $file | grep '^ 0.00000  0.00000  0.00000' | wc -l | xargs`
    if [ '$count' == '' ]; then
        color=`gs -q -o - -sDEVICE=inkcov $file | grep -v '^ 0.00000  0.00000  0.00000' | wc -l | xargs`
    else
        color=`expr $count - $bw`
    fi
    echo "B&W:        $bw pages"
    echo "Color:      $color pages" 
else
    echo "Color information unavailable - requires tool 'gs'."
fi
