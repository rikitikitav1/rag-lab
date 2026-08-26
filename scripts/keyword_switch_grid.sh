set -e
out_dir=/app/datasets/measurements/grid
mkdir -p "${out_dir#/app/}"
for s in curated paraphrased paraphrased_ru; do
  for q in and or; do
    for r in ts_rank ts_rank_cd; do
      for n in 0 2; do
        for l in langdetect cyrillic_ratio; do
          tag="${s}__${q}_${r}_n${n}_${l}"
          res=$(docker compose exec -T -e PYTHONPATH=/app/app worker python /app/scripts/retrieval_report.py \
            --set "$s" --keyword-query "$q" --keyword-rank "$r" --keyword-norm "$n" \
            --query-lang "$l" --out "$out_dir/$tag.json" 2>/dev/null | grep -E "^  (file|section)")
          f=$(echo "$res" | grep file | grep -o '"MRR@20": [0-9.]*' | cut -d' ' -f2)
          sec=$(echo "$res" | grep section | grep -o '"MRR@20": [0-9.]*' | cut -d' ' -f2)
          printf "%-15s %-4s %-11s norm=%s %-14s file=%-7s section=%s\n" \
            "$s" "$q" "$r" "$n" "$l" "$f" "${sec:-n/a}"
        done
      done
    done
  done
done
