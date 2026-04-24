# 1. 实例化
input_dim = 1 # 假设 x 是单维度时间序列
num_harmonics = 5
encoder = VariationalIndependentTimeSeriesTransformer(
    input_dim=input_dim, 
    output_dim=num_harmonics,
    hidden_dim=128
)
model = PhysicalHarmonicVAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. 训练步
model.train()
for epoch in range(epochs):
    for x_batch, t_batch in dataloader:
        x_batch = x_batch.to(device) # (batch, seq, 1)
        t_batch = t_batch.to(device) # (batch, seq)

        optimizer.zero_grad()
        
        # 前向传播
        x_hat, dist_params = model(x_batch, t_batch)
        
        # 计算 Loss
        loss, recon, kl = compute_harmonic_elbo(x_batch, x_hat, dist_params)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪 (由于 Transformer 和复杂的指数函数，推荐加入裁剪防爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()