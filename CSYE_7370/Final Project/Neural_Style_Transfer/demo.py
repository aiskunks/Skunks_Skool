import scipy.misc
import tensorflow as tf

from utils import compute_content_cost, compute_style_cost, total_cost, generate_noise_image, save_image, \
    reshape_and_normalize_image, load_vgg_model


def transfer(num_iterations=200):
    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    # Initialize a noisy image by adding random noise to the content_image
    generated_image = generate_noise_image(content_image)

    # load VGG19 model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(sess, model)

    J = total_cost(J_content, J_style, alpha, beta)  # 10,40

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(generated_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model["input"])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("images/" + str(i) + ".png", generated_image)

    # save last generated image

    save_image("images/output.png", generated_image)

    return generated_image


if __name__ == '__main__':
    # Parameters
    alpha = 10
    beta = 40
    iterations = 200

    content_image = scipy.misc.imread("images/content.jpg")
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread("images/style.jpg")
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image)

    transfer()
