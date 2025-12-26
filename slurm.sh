for n in $(sinfo -p ghx4 -h -N -o %N); do
  scontrol show node -o "$n"
done | awk '
{
  node=""; cfg=0; alloc=0;
  if (match($0, /NodeName=([^ ]+)/, a)) node=a[1];
  if (match($0, /CfgTRES=[^ ]*gres\/gpu=([0-9]+)/, a)) cfg=a[1];
  if (match($0, /AllocTRES=[^ ]*gres\/gpu=([0-9]+)/, a)) alloc=a[1];
  free=cfg-alloc;
  if (cfg>0) printf "%s free_gpu=%d (alloc=%d/%d)\n", node, free, alloc, cfg;
}' | sort -k2,2nr 