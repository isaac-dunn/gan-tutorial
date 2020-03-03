# AIMS GAN Lab Sheet

## Step 0: Get set up
- Clone this repository.
- Install the requirements (`torch`, `matplotlib`, `numpy`) - sensible to do it in a virtual environment.
- Make sure that importing `tkinter` works - you may need to install it if it doesn’t.

Here is a link to the lecture slides for reference: https://drive.google.com/open?id=1c5KA1m1bOtmralSB-G1ufT18s_Z3yB7k

## Step 1: Implement a plain GAN
Use `gan_template.py` as a template.

 - Define a simple feed-forward architecture for the generator and the discriminator.
 - Remember that the generator needs to generate 2D points, and the discriminator needs to output a probability of each example being real.
 - In the training loop, use your generator to generate a batch of ‘fake’ data.
 - Define the GAN loss functions.

Hint: you may find `torch.nn.functional.binary_cross_entropy` helpful.

Hint: you may want to `.detach()` the generated data when using it to compute the discriminator loss.

## Step 2: Observe training
Run the program. Observe whether the GAN is able to learn to fit the data.

Experiments you may like to try:
 - What happens if you change the training distribution? Vary its mean, standard deviation, and number of modes.
 - What happens if you use the defined `weights_init()` function as opposed to using the default initialisation?
 - What happens if you vary your network architecture?
 - What happens if you vary the learning rates?

## Step 3: Try the alternative generator loss
In the original GAN paper, the authors suggest that it may be better to train the generator to minimise `-log(d(g(z)))` rather than to minimise `log(1-d(g(x)))`.

Why might this make training easier? Does it make a noticeable difference?

## Step 4: Implement the Wasserstein GAN
Either with weight clipping or with the gradient penalty.

For weight clipping you'll need to `apply` a function to the discriminator that `clamps` its `weight.data` values.

For the gradient penalty, give it a go just from the definition. Or if you need a hint, look at `grad_penalty_hint.py`.


## Step 5: Implement a GAN for MNIST
Use `mnist_gan_template.py` as a starting point - you may like to copy some of your training code into the loop.

You'll need to either change your generator and discriminator to work with images of shape (1, 28, 28), or else define new networks entirely.

You may want to use a convolutional generator and/or discriminator. Here is one possibility for the generator (may work without the batch norm and dropout):
![Generator architecture](https://i.imgur.com/yWC6Tmt.png)
