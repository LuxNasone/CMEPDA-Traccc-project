    if [ -f "$reco_out/performance.csv" ]; then
        cp "$reco_out/performance.csv" "$results/${backend}_${dataset}.csv"
        # The tracks of every event, for the efficiency studies
        tracks="$results/${backend}_${dataset}_tracks"
        rm -rf "$tracks"
        mkdir -p "$tracks"
        find "$reco_out" -maxdepth 1 -name 'event*-track*.csv' -exec cp {} "$tracks"/ \;
        echo "Saved: $results/${backend}_${dataset}.csv, $log and $tracks/"
    else
        echo "No $reco_out/performance.csv found: set RECO_OUT to the folder" \
             "that contains reco_${backend}" >&2
    fi
done
