if mask is None:
    with torch.no_grad():
        mask_shape = [bs, 1, max_len, max_len]
        mask = torch.ones(mask_shape, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)