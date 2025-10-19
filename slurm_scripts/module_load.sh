mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

# 激活时：自动 load（把 cuda/12.9 换成你要的模块，可空格分隔多个）
cat > "$CONDA_PREFIX/etc/conda/activate.d/01-modules.sh" <<'EOF'
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module load cuda-compat/12.9
EOF

# 退出时：自动 unload（与上面保持一致）
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/01-modules.sh" <<'EOF'
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module unload cuda-compat/12.9
EOF