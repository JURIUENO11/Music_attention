import tensorflow as tf

def AAD(shape_eeg, shape_sti,
        kernels=32, kernel_size=7, strides=1, dilation_rate=3,
        units=48,
        size=16, sigma=0.7,
        sources=4):
    """ Parameters:
        shape_eeg:     tuple, shape of EEG (sampling_points_num, channel_num), e.g., (256, 64) when EEG has 64 channels, and the sampling rate of a 2-seconds EEG segment is 128 Hz.
        shape_sti:     tuple, shape of stimulus (sampling_points_num, feature_dim) or (sampling_points_num, feature_dim, feature_num), e.g., (256, 24, 2) when each of two audio feature has 24 dimensions, and two audio features need to be fused for each sound source. 
        kernels:       int, number of output filters in the 1D convolution
        kernel_size:   int, length of the 1D convolution window
        strides:       int, stride length of the 1D convolution
        dilation_rate: int, dilation rate to use for the dilated 1D convolution
        units:         int, dimensionality of the output space of the GRU layer
        size:          int, size of gaussian filter of the SSIM index
        sigma:         int, width of gaussian filter of the SSIM index
        sources:       int, number of sound sources in a mixed stimulus
    """
    # 1) Inputs
    input0 = tf.keras.layers.Input(shape=shape_eeg);    eeg = input0 # EEG or CSP-filtered EEG
    input1 = tf.keras.layers.Input(shape=shape_sti);    sti1 = input1 # Audio eature of the 1st sound source within a mixed stimulus
    input2 = tf.keras.layers.Input(shape=shape_sti);    sti2 = input2 # 2nd sound source
    input3 = tf.keras.layers.Input(shape=shape_sti);    sti3 = input3 # 3rd sound source
    input4 = tf.keras.layers.Input(shape=shape_sti);    sti4 = input4 # 3rd sound source

    # 2) Path for EEG
    eeg = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)(eeg)
    eeg = tf.keras.layers.BatchNormalization()(eeg)
    # eeg = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(eeg) # No pooling, unless out of GPU memory
    eeg = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)(eeg)

    # 3) Path for stimulus
    if len(sti1.shape) == 4:    # The 4th value of the stimulus shape is feature_num, e.g., feature_num=2 represents the fusion of two audio features here
        feature_fusion = tf.keras.layers.Dense(1, activation=None, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg()) # The weights cannot be negative
        sti1 = feature_fusion(sti1);    sti1 = tf.squeeze(sti1, -1)
        sti2 = feature_fusion(sti2);    sti2 = tf.squeeze(sti2, -1)
        sti3 = feature_fusion(sti3);    sti3 = tf.squeeze(sti3, -1)
        sti4 = feature_fusion(sti4);    sti4 = tf.squeeze(sti4, -1)

    Conv1D = tf.keras.layers.Conv1D(kernels, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None)
    sti1 = Conv1D(sti1)
    sti2 = Conv1D(sti2)
    sti3 = Conv1D(sti3)
    sti4 = Conv1D(sti4)

    BN = tf.keras.layers.BatchNormalization()
    sti1 = BN(sti1)
    sti2 = BN(sti2)
    sti3 = BN(sti3)
    sti4 = BN(sti4)   
    # Pooling = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2) # Same as above
    # sti1 = Pooling(sti1)
    # sti2 = Pooling(sti2)
    # if sources == 3:
    #     sti3 = Pooling(sti3)

    GRU = tf.compat.v1.keras.layers.CuDNNGRU(units, return_sequences=True)
    sti1 = GRU(sti1)
    sti2 = GRU(sti2)
    sti3 = GRU(sti3)
    sti4 = GRU(sti4)
    # 4) Classification
    # ※ Cosine similarity has been discarded!
    # Dot1 = tf.keras.layers.dot([eeg, sti1], 2, normalize=True)
    # CS1 = tf.math.reduce_mean(tf.linalg.diag_part(Dot1), axis=-1);    CS1 = tf.expand_dims(CS1, axis=-1)
    # Dot2 = tf.keras.layers.dot([eeg, sti2], 2, normalize=True)
    # CS2 = tf.math.reduce_mean(tf.linalg.diag_part(Dot2), axis=-1);    CS2 = tf.expand_dims(CS2, axis=-1)
    # if sources == 3:
        # Dot3 = tf.keras.layers.dot([eeg, sti3], 2, normalize=True)
        # CS3 = tf.math.reduce_mean(tf.linalg.diag_part(Dot3), axis=-1);    CS3 = tf.expand_dims(CS3, axis=-1)
        
    # ※※ SSIM
    # min_len = tf.minimum(tf.shape(eeg)[1], tf.shape(sti1)[1])
    # eeg = eeg[:, :min_len, :]
    # sti1 = sti1[:, :min_len, :]
    # sti2 = sti2[:, :min_len, :]
    # sti3 = sti3[:, :min_len, :]
    # sti4 = sti4[:, :min_len, :]
    eeg = tf.expand_dims(eeg, -1);  sti1 = tf.expand_dims(sti1, -1);    sti2 = tf.expand_dims(sti2, -1)
    SSIM1 = tf.image.ssim(eeg, sti1, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM1 = tf.expand_dims(SSIM1, axis=-1)
    SSIM2 = tf.image.ssim(eeg, sti2, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM2 = tf.expand_dims(SSIM2, axis=-1)
    sti3 = tf.expand_dims(sti3, -1)
    SSIM3 = tf.image.ssim(eeg, sti3, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM3 = tf.expand_dims(SSIM3, axis=-1)
    sti4 = tf.expand_dims(sti4, -1)
    SSIM4 = tf.image.ssim(eeg, sti4, max_val=2, filter_size=size, filter_sigma=sigma)
    SSIM4 = tf.expand_dims(SSIM4, axis=-1)

    SSIM_concat = tf.keras.layers.concatenate([SSIM1, SSIM2, SSIM3, SSIM4])

    one_hot = tf.keras.layers.Dense(units=sources, activation='softmax')(SSIM_concat)
    
    # Building a model

    model = tf.keras.Model(inputs=[input0, input1, input2,input3,input4], outputs=one_hot)
    
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['sparse_categorical_accuracy'])
    return model
