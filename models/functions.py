from torch.autograd import Function

class ReverseLayerF(Function):
    '''
    Reverse layer for DANN.
    We don't add alpha to weight the gredient from discriminator here.
    We add the alpha to the loss.
    '''
    @staticmethod
    def forward(ctx, x):
        # ctx save for backward
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None
