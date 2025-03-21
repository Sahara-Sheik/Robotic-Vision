import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import torch



# We will use a latent_dim of 128 for the Conv-VAE
LATENT_DIM = 128

# ------------------------------------------------------------------------
            # DATA LOADING
# ------------------------------------------------------------------------
def load_dataset(main_folder):
    """
    Walk through subfolders, detect ArUco in images, and read rc-position-target for these 6 parameters:
      (1) height, (2) distance, (3) rotation->heading, (4) wrist_angle, (5) wrist_rotation, (6) gripper.

    Returns three  lists (all same length):
        X_data: list of ArUco feature vectors (2D corners).
        Y_data: list of 6D targets.
        images_list: list of raw images (np.array shape (H, W, 3)).
    """
    # Prepare ArUco detection (no camera calibration needed)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters()



    # 2. Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    #  Load an image (BGR format)
    # frame = cv.imread("my pics")
    # Detect markers in the image, new Aruco 4 librarary
    # markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)


    X_data = []
    Y_data = []
    images_list = []
    counter =0
    match = 0
    # Traverse each subfolder
    for subfolder in os.listdir(main_folder):
        sub_path = os.path.join(main_folder, subfolder)
        # print(sub_path)
        if not os.path.isdir(sub_path):
            continue

        # For each file in the subfolder, look for images + matching JSON
        for fname in os.listdir(sub_path):
            counter=counter+1
            # print(counter)
            f_lower = fname.lower()
            if f_lower.endswith((".jpg", ".jpeg", ".png")):
                # Found an image
                image_path = os.path.join(sub_path, fname)
                base_name, _ = os.path.splitext(fname)
                # print(base_name)
                # print(image_path)
                cleaned_base_name = base_name.replace("_dev2", "")


                # Matching JSON
                json_path = os.path.join(sub_path, cleaned_base_name + ".json")
                # print(json_path)
                # if not os.path.isfile(json_path):
                #     print("nooooooooooooooooooooooooooooooooooooooooo")

                #     continue

                # Load JSON data
                if os.path.isfile(json_path):
                    match = match+1
                    # print(match)
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        # print("Loaded JSON Data:")

                        # print(json.dumps(data, indent=4))

                    # Extract 6 parameters from "rc-position-target"
                    pos_data = data["rc-position-target"]
                    height = float(pos_data["height"])
                    distance = float(pos_data["distance"])
                    heading = float(pos_data["rotation"])  # rename "rotation" -> "heading"
                    wrist_angle = float(pos_data["wrist_angle"])
                    wrist_rotation = float(pos_data["wrist_rotation"])
                    gripper = float(pos_data["gripper"])
                    y_target = [height, distance, heading, wrist_angle, wrist_rotation, gripper]
                    # print(y_target)

                    # Load the image
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is None:
                        print("yasssssssssssssssssssssssssssssssssss")
                        continue

                    # Detect ArUco markers in 2D
                    corners, ids, _ = detector.detectMarkers(image_bgr)
                    # print(ids)
                    # print("cornersssss")
                    if ids is None or len(ids) == 0:
                        corners =(np.array([[[94., 68.],
                            [94., 76.],
                            [88., 77.],
                            [88., 68.]]], dtype=np.float32),)
                        # If no markers found, skip
                        # continue
                    # print("found at lease one marker")
                    # Flatten corners -> feature vector
                    features = []


                    for c_set in corners:
                        # print(corners)
                        # print(c_set)
                        for (x, y) in c_set[0]:
                            # print(x,y)
                            features.extend([x, y])

                    # Append results
                    X_data.append(features)
                    Y_data.append(y_target)
                    images_list.append(image_bgr)  # keep the raw image for VGG19 / VAE

    # Convert lists to arrays
    X_array = np.array(X_data, dtype=np.float32)
    Y_array = np.array(Y_data, dtype=np.float32)
    return X_array, Y_array, images_list

# ------------------------------------------------------------------------
#               CONV-VAE MODEL (with latent size = 128)
# ------------------------------------------------------------------------
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], LATENT_DIM))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_regressor(encoder, latent_dim=128, output_dim=6):
    """
    Creates a Keras Model:
      Input: (image_height, image_width, 3)
      Output: 6D parameters
    Uses the *pre-trained, frozen* encoder from the VAE,
    followed by a small MLP to map latent->6D.
    """
    # Freeze encoder to preserve the learned latent space
    encoder.trainable = False

    # Define new model that calls the encoder first
    image_input = keras.Input(shape=(64, 64, 3), name="vae_regressor_input")
    z_mean, z_log_var, z = encoder(image_input, training=False)
    # We'll use z (the sampled latent) or z_mean—both can work.
    # I read z_mean tends to be more stable. Let’s pick z_mean here:
    latent_vector = z_mean

    # MLP on top of the 128-D latent
    x = layers.Dense(64, activation='relu')(latent_vector)
    x = layers.Dropout(0.2)(x)
    output_6d = layers.Dense(output_dim, activation='linear')(x)

    regressor_model = keras.Model(inputs=image_input, outputs=output_6d, name='vae_regressor')
    regressor_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    return regressor_model


def build_conv_vae(image_height, image_width, channels=3):
    """
    Build a Convolutional VAE with:
      - latent_dim = 128
      - Convolutional encoder
      - Convolutional decoder
    Returns: (vae_model, encoder, decoder)
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(image_height, image_width, channels))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    y = layers.Dense(8*8*128, activation='relu')(latent_inputs)
    y = layers.Reshape((8, 8, 128))(y)
    y = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(y)
    y = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(y)
    y = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(y)
    # Final layer: produce output of shape (image_height, image_width, channels)
    decoder_outputs = layers.Conv2DTranspose(
        channels, 3, padding='same', activation='sigmoid'
    )(y)
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    # Custom VAE class
    class VAE(keras.Model):
        def __init__(self, enc, dec, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = enc
            self.decoder = dec

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]  # only use images
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z_ = self.encoder(data, training=True)
                reconstruction = self.decoder(z_, training=True)
                # 1) Reconstruction Loss
                recon_loss = tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=[1,2]  # sum over height/width
                )
                # 2) KL Divergence
                kl_loss = -0.5 * tf.reduce_sum(
                    (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1
                )
                total_loss = tf.reduce_mean(recon_loss + kl_loss)

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "recon_loss": tf.reduce_mean(recon_loss),
                "kl_loss": tf.reduce_mean(kl_loss),
            }

        def call(self, inputs, training=False):
            z_mean, z_log_var, z_ = self.encoder(inputs, training=training)
            return self.decoder(z_, training=training)

    vae_model = VAE(encoder, decoder)
    vae_model.compile(optimizer=keras.optimizers.Adam())
    return vae_model, encoder, decoder

# ------------------------------------------------------------------------
#               VGG19-BASED REGRESSOR
# ------------------------------------------------------------------------

def build_vgg19_regressor(num_outputs=6):
    """
    Build a Keras model using VGG19 (ImageNet weights) as a feature extractor,
    then add a small regression head for `num_outputs` parameters (e.g. 6D).
    """
    # Load VGG19 backbone
    # Use include_top=False to remove the default classification head
    # We'll freeze these layers by default
    base_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze VGG19 layers
    for layer in base_vgg.layers:
        layer.trainable = False

    # Build a custom top
    x = base_vgg.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_outputs, activation='linear')(x)

    model = keras.Model(inputs=base_vgg.input, outputs=outputs, name="VGG19_Regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='mse',
        metrics=['mae']
    )
    return model

# ------------------------------------------------------------------------
#               MAIN PIPELINE
# ------------------------------------------------------------------------
def main():
    # 1) LOAD DATASET
    main_folder = "/home/ssheikholeslami/BerryPicker2/BlueBerry"  # <-- update if we change to a central exoeriment
    X, Y, images_list = load_dataset(main_folder)
    print("Dataset loaded.")
    print("  # of ArUco-based samples:", len(X))
    print("  Feature shape (X):", X.shape if len(X) else None)
    print("  Target shape (Y):", Y.shape if len(Y) else None)
    print("  # of raw images for VAE/VGG19:", len(images_list))

    # --------------------------------------------------------------------
    #       TRY MODELS
    # --------------------------------------------------------------------
    if len(X) < 5:
        print("Not enough data to train RF models. Skipping.")
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        # RF did not work at all flat, why?
        # RandomForest = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=101)
        # RandomForest.fit(X_train, Y_train)
        # pred_RandomForest = RandomForest.predict(X_test)

        #plot saver
    def save_prediction_plot(N ,Y_test, pred_conv_vae, pred_test_vgg19,  save_path="results_plot.png"):
        """
        Saves a comparison plot of ground truth vs predictions for different models.

        Parameters:
        - Y_test: Ground truth values (test set)
        - pred_RandomForest: Predictions from the RandomForest model
        - pred_test_vgg19: Predictions from the VGG19-based model
        - pred_conv_vae: Predictions from the pred_conv_vae-based model
        - save_path: Path where the plot image will be saved
        """

        param_names = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]

        # Choose how many test samples to visualize
        # N = min(100, len(Y_test))
        Y_true_slice = Y_test[:N]
        pred_rf_slice = pred_conv_vae[:N]
        pred_vgg_slice = pred_test_vgg19[:N]
        # pred_aruco_slice = pred_aruco[:N]

        plt.figure(figsize=(12, 8))

        for i, pname in enumerate(param_names):
            plt.subplot(2, 3, i + 1)
            plt.plot(range(N), Y_true_slice[:, i], color='blue', label='groun truth')
            plt.plot(range(N), pred_rf_slice[:, i], color='orange', label='conv-vae-128')
            plt.plot(range(N), pred_vgg_slice[:, i], color='green', label='VGG19')
            # plt.plot(range(N), pred_aruco_slice[:, i], color='red', label='CNN')

            plt.title(pname)
            plt.ylim([-180, 180])  # Adjust this based on your data range
            if i == 0:
                plt.legend(loc='upper right')

        plt.tight_layout()

        # Save the figure instead of showing it
        plt.savefig(save_path, dpi=300)
        plt.close()  # Close to avoid inline display issues

        print(f"Plot saved to {os.path.abspath(save_path)}")

        # # "mlp-vgg19-128" (placeholder)
        # model_vgg19_rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=202)
        # model_vgg19_rf.fit(X_train, Y_train)
        # pred_vgg19_rf = model_vgg19_rf.predict(X_test)

        # # "aruco-128" (placeholder)
        # model_aruco = RandomForestRegressor(n_estimators=120, max_depth=15, random_state=303)
        # model_aruco.fit(X_train, Y_train)
        # pred_aruco = model_aruco.predict(X_test)

        # Quick plot: Compare predictions for first N test samples
        # param_names = ["height", "distance", "heading", "wrist_angle", "wrist_rotation", "gripper"]
        # N = min(50, len(X_test))
        # Y_true_slice = Y_test[:N]
        # pred_conv_slice = pred_conv_vae[:N]
        # pred_vgg_slice = pred_vgg19_rf[:N]
        # pred_aruco_slice = pred_aruco[:N]

        # plt.figure(figsize=(12, 8))
        # for i, pname in enumerate(param_names):
        #     plt.subplot(2, 3, i + 1)
        #     plt.plot(range(N), Y_true_slice[:, i], color='blue', label='truth')
        #     plt.plot(range(N), pred_conv_slice[:, i], color='orange', label='mlp-conv-vae-128')
        #     plt.plot(range(N), pred_vgg_slice[:, i], color='green', label='mlp-vgg19-128')
        #     plt.plot(range(N), pred_aruco_slice[:, i], color='red', label='aruco-128')
        #     plt.title(pname)
        #     plt.ylim([-180, 180])  # adapt to your data range
        #     if i == 0:
        #         plt.legend(loc='upper right')
        # plt.tight_layout()
        # plt.show()



    # --------------------------------------------------------------------
    # CONV-VAE (Unsupervised reconstruction with latent=128)
    # --------------------------------------------------------------------



    if len(images_list) < 5:
        print("Not enough images to train a Conv-VAE. Skipping.")
    else:
        # Convert all raw images to (64, 64) for the VAE
        target_h, target_w = 64, 64
        images_vae = []
        for img_bgr in images_list:
            #  resize
            resized = cv2.resize(img_bgr, (target_w, target_h))
            # scale to [0..1]
            resized = resized.astype('float32') / 255.0
            images_vae.append(resized)
        images_vae = np.array(images_vae)
        print("Training Conv-VAE on shape:", images_vae.shape)

        vae, encoder, decoder = build_conv_vae(target_h, target_w, channels=3)
        vae.fit(images_vae, epochs=100, batch_size=8)  # just a small demo

        #  reconstructing the first image
        test_img = images_vae[0:1]
        reconstructed = vae.predict(test_img)

        plt.figure(figsize=(4,2))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(test_img[0][:,:,::-1])  # BGR->RGB
        plt.subplot(1,2,2)
        plt.title("Reconstruction")
        plt.imshow(reconstructed[0][:,:,::-1])  # also BGR->RGB
        plt.show()

        vae_regressor = build_vae_regressor(encoder, latent_dim=128, output_dim=6)



        # Create a (64x64) version of your images for the regressor
        X_vae_train = []
        target_h, target_w = 64, 64
        for img_bgr in images_list:
            resized = cv2.resize(img_bgr, (target_h, target_w))
            resized = resized.astype('float32') / 255.0
            X_vae_train.append(resized)
        X_vae_train = np.array(X_vae_train)

        # Y is your 6D array. Make sure it's a NumPy array of shape (num_samples, 6).
        # Y_6d = np.array(Y)

        # Split into train/test sets if desired
        X_train_img, X_test_img, Y_train_6d, Y_test_6d = train_test_split(
            images_vae, Y, test_size=0.3, random_state=123
        )

        # Train the regressor on the 6D
        history = vae_regressor.fit(
            X_train_img, Y_train_6d,
            validation_data=(X_test_img, Y_test_6d),
            epochs=100,
            batch_size=8
        )
        pred_conv_vae = vae_regressor.predict(X_test_img)

    # --------------------------------------------------------------------
    #       VGG19-BASED REGRESSOR (Direct image -> 6D target)
    # --------------------------------------------------------------------
    # We need a consistent list of images + matching 6D Y.
    # Because we only appended to images_list when corners were found,
    # X, Y, images_list all have the same length. Let's use them here:
    if len(images_list) < 5:
        print("Not enough images for VGG19 training. Skipping.")
    else:
        # Convert images for VGG19: (224, 224), scale, plus we need Y for the same samples
        X_imgs = []
        for img_bgr in images_list:
            resized = cv2.resize(img_bgr, (224,224))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Convert to float & use VGG19 preprocess
            resized = resized.astype('float32')
            # Preprocess for VGG19
            resized = preprocess_input(resized)  # subtract mean, etc.
            X_imgs.append(resized)

        X_imgs = np.array(X_imgs)
        Y_6d = np.array(Y)  # same shape as random forest Y

        print("Training VGG19-based regressor. X_imgs shape:", X_imgs.shape, "Y_6d:", Y_6d.shape)

        # Train-test split
        X_train_img, X_test_img, Y_train_img, Y_test_img = train_test_split(
            X_imgs, Y_6d, test_size=0.3, random_state=999
        )

        vgg19_reg = build_vgg19_regressor(num_outputs=6)
        vgg19_reg.summary()

        #  training -> ( tune epochs, batch_size)
        history = vgg19_reg.fit(
            X_train_img, Y_train_img,
            validation_data=(X_test_img, Y_test_img),
            epochs=100,
            batch_size=8
        )

        # Predict on test set
        pred_test_vgg19 = vgg19_reg.predict(X_test_img)

        # Plot one parameter (e.g. 'height') to see how well it does
        plt.figure(figsize=(6,4))
        plt.plot(Y_test_img[:,0], label='true height', color='blue')
        plt.plot(pred_test_vgg19[:,0], label='pred height', color='orange')
        plt.title("VGG19 Regressor: 'height' predictions")
        plt.legend()
        plt.show()

        # Evaluate on all 6D
        mse = np.mean((pred_test_vgg19 - Y_test_img)**2, axis=0)
        print("VGG19 MSE per dimension:", mse)
        print("VGG19 overall MSE:", np.mean(mse))

    save_prediction_plot(100, Y_test,  pred_conv_vae, pred_test_vgg19,  save_path="results_plot100.png")
    save_prediction_plot(50, Y_test,  pred_conv_vae, pred_test_vgg19,  save_path="results_plot50.png")
if __name__ == "__main__":
    main()

