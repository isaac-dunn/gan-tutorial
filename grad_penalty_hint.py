def compute_gradient_penalty(batch, gen_batch, d):
    batch_size = batch.shape[0]
    alpha = torch.rand(batch_size,1)
    interpolates = # TODO use alpha to interpolate between real and fake
    d_on_interpolates = d(interpolates)
    gradients = torch.autograd.grad(
            outputs=#TODO: the expression being differentiated
            inputs=#TODO: the outputs are differentiated wrt inputs
            grad_outputs=torch.ones(batch_size, 1),
        )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = #TODO: mean of (||grads||_2 - 1)^2
    return gradient_penalty

