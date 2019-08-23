awk -F'\t' '!_[$1]++ { 
  fn && close(fn)
  fn = $1 ".tab"
  }
{ print > fn }
' enc_output.tsv
