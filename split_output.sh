awk -F'\t' '!_[$3]++ { 
  fn && close(fn)
  fn = $3 ".tab"
  }
{ print > fn }
' encout_stripped.tsv
