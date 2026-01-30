#!/bin/bash

echo "=== Enter commit TITLE (Subject)"
read -r TITLE

if [ -z "$TITLE" ]; then
  echo "Error: Commit title cannot be empty"
  exit 1
fi

echo "=== Enter commit DESCRIPTION"
echo "(Press Enter twice on an empty line to finish)"

DESCRIPTION=""
while IFS= read -r line; do

    if [ -z "$line" ]; then
        break
    fi

    DESCRIPTION+="$line"$'\n'
done

git commit -m "$TITLE" -m "$DESCRIPTION"

echo "---"
echo "Done! Commit successful."