# Efficient-Multi-Collection-Style-Transfer-Using-GAN

Principles of Deep Learning Final Project, University of Pennsylvania

Proposed a new model that can make style transfer from single style image, and allow to transfer into multiple different styles in a single model.

Applied patch permutation and adversarial gated networks for multi-collection single-style image style transfer.

From the experiment, we noticed that there are mainly two issues with the transferred image. First of all, some content of the image are lost after transferring to a style, and we will consider to construct a convolutional network for classification. On the other hand, the transferred image have many grids after training a large number of epochs. We will try to do patch permutation on transferred images before feeding into the discriminative network or edit loss to figure out the reason of this issue.
