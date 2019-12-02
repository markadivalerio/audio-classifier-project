#!/bin/bash

SAMPLE_RATE=22050

# fetch_clip(videoID, startTime, endTime)
fetch_clip() {
  echo "Fetching $1 ($2 to $3)..."
  outname="$1_$2"
  if [ -f "${outname}.wav.gz" ]; then
    echo "Already have it."
    return
  fi

  youtube-dl https://youtube.com/watch?v=$1 --quiet --extract-audio --audio-format wav --output "./youtube/$outname.%(ext)s"
  if [ $? -eq 0 ]; then
    # If we don't pipe `yes`, ffmpeg seems to steal a
    # character from stdin. I have no idea why.
    yes | ffmpeg -loglevel quiet -i "./youtube/$outname.wav" -ar $SAMPLE_RATE \
      -ss "$2" -to "$3" "./youtube/${outname}_out.wav"
    mv "./youtube/${outname}_out.wav" "./youtube/$outname.wav"
    gzip "./$outname.wav"
  else
    # Give the user a chance to Ctrl+C.
    sleep 1
  fi
}

grep -E '^[^#]' | while read line
do
  #  fetch_clip $(echo "$line" | sed -E 's/, / /g')
  fetch_clip $(echo "$line" | awk -F "," '{print $1,$2,$3}')
done
