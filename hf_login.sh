#!/usr/bin/env bash

MYTOKEN="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"

if [ "$1" ]; then
    MYTOKEN=$1
fi
huggingface-cli login --token $MYTOKEN