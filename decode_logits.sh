#!/usr/bin/env bash
#
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2020  Alex Hung (hung_alex@icloud.com)
# Apache 2.0

cmd=run.pl

beam=17.0
acwt=1.0
post_decode_acwt=10.0 # see https://kaldi-asr.org/doc/chain.html#chain_decoding
max_active=7000
min_active=200
max_mem=200000000 # approx. limit to memory consumption during minimization in bytes
lattice_beam=8.0  # Beam we use in lattice generation.

#. parse_options.sh || exit 1;

lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >lat.gz"

symbols=data/chars_new_symbol_table.txt
hyps_file_tmp=val_hyps_tmp
hyps_file=val_hyps
latgen-faster --word-symbol-table=$symbols \
  --allow-partial=true \
  --acoustic-scale=$acwt \
  --max-mem=$max_mem --min-active=$min_active --max-active=$max_active \
  --beam=$beam --lattice-beam=$lattice_beam \
  data/T.fst ark:logits.ark "$lat_wspecifier"

wip=0.0
lattice-scale --acoustic-scale=10 --lm-scale=10 "ark:gunzip -c lat.gz|" ark:- | lattice-add-penalty \
  --word-ins-penalty=$wip ark:- ark:- | lattice-best-path \
  --word-symbol-table=data/chars_new_symbol_table.txt ark:- ark,t: | int2sym.pl -f 2- $symbols >${hyps_file_tmp}

python preprocess_texts.py ${hyps_file_tmp} ${hyps_file} --reverse --in-uttids --out-uttids

python score.py ${hyps_file} data/val_texts
