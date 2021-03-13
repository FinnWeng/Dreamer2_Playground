import numpy as np
import tensorflow as tf


class Attn_Head(tf.keras.layers.Layer):
    def __init__(
        self, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False
    ):
        super(Attn_Head, self).__init__()

        self.bias_mat = bias_mat
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.activation = activation
        self.dropout = tf.keras.layers.Dropout(1.0 - in_drop)
        self.conv1 = tf.keras.layers.Conv1D(out_sz, 1, use_bias=False)
        self.conv2 = tf.keras.layers.Conv1D(1, 1)
        self.conv3 = tf.keras.layers.Conv1D(1, 1)
        self.coef_dropout = tf.keras.layers.Dropout(1.0 - coef_drop)
        self.bias_add_layer1 = tf.keras.layers.Dense(
            out_sz
        )  # I'm lazy. So just use Dense to add bias

        res_conv = tf.keras.layers.Conv1D(out_sz, 1)

    def __call__(self, seq):
        if self.in_drop != 0.0:
            seq = self.dropout(seq)
        seq_fts = self.conv1(seq)

        # simplest self-attention possible
        f_1 = self.conv2(seq_fts)
        f_2 = self.conv3(seq_fts)
        print(f_1)
        print(tf.transpose(f_2, [0, 2, 1]))
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # matrix[seq.shape[1]*seq.shape[1]]
        # print(logits)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat)
        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_drop != 0.0:
            seq_fts = self.dropout(seq_fts)

        vals = tf.linalg.matmul(coefs, seq_fts)  # filter :out_sz
        print("vals:", vals)

        ret = self.bias_add_layer1(vals)

        # residual connection
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + res_conv(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return self.activation(ret)  # activation


class GAT(tf.keras.Model):
    # nb_nodes = features.shape[0] => for my 9
    # ft_size = features.shape[1] => for my 6
    # nb_classes = y_train.shape[1]  => for my 2 (since 6*2 near 16 for conv encdoer)
    # n_heads = [8, 1]
    # hid_units = [8]
    def __init__(
        self,
        nb_classes,
        nb_nodes,
        training,
        attn_drop,
        ffd_drop,
        bias_mat,
        hid_units,
        n_heads,
        activation=tf.nn.elu,
        residual=False,
    ):
        self.n_heads = n_heads
        super(GAT, self).__init__()

        self.attns_ins = []
        for _ in range(n_heads[0]):
            self.attns_ins.append(
                Attn_Head(
                    bias_mat=bias_mat,
                    out_sz=hid_units[0],
                    activation=activation,
                    in_drop=ffd_drop,
                    coef_drop=attn_drop,
                    residual=False,
                )
            )

        self.attn_blocks = []
        for i in range(1, len(hid_units)):
            print("i:", i)
            attn_hid = []
            for _ in range(n_heads[i]):
                attn_hid.append(
                    Attn_Head(
                        bias_mat=bias_mat,
                        out_sz=hid_units[i],
                        activation=activation,
                        in_drop=ffd_drop,
                        coef_drop=attn_drop,
                        residual=residual,
                    )
                )
                self.attn_blocks.append(attn_hid)

        self.outs = []
        for i in range(n_heads[-1]):
            self.outs.append(
                Attn_Head(
                    bias_mat=bias_mat,
                    out_sz=nb_classes,
                    activation=lambda x: x,
                    in_drop=ffd_drop,
                    coef_drop=attn_drop,
                    residual=False,
                )
            )

    def __call__(self, inputs):
        attns = []
        for attn in self.attns_ins:
            temp = attn(inputs)
            print("temp:", temp.shape)
            attns.append(attn(inputs))
            # attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
            #     out_sz=hid_units[0], activation=activation,
            #     in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        # for i in range(1, len(hid_units)):
        #     h_old = h_1
        #     attns = []
        #     for _ in range(n_heads[i]):
        #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #             out_sz=hid_units[i], activation=activation,
        #             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        #     h_1 = tf.concat(attns, axis=-1)

        for attn_block in self.attn_blocks:
            attns = []
            for attn in attn_block:
                attns.append(attn(h_1))

        out = []
        # for i in range(n_heads[-1]):
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #         out_sz=nb_classes, activation=lambda x: x,
        #         in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        for attn in self.outs:
            temp = attn(h_1)
            print("temp:", temp.shape)
            out.append(attn(h_1))
        logits = tf.add_n(out) / self.n_heads[-1]

        return logits


if __name__ == "__main__":

    # test Attn_Head

    bias_mat = np.arange(25, dtype=np.float32)
    bias_mat = bias_mat.reshape([1, 5, 5])
    attn_head = Attn_Head(10, bias_mat, tf.nn.relu)
    a = np.array([[[1], [1], [1], [1], [1]]], dtype=np.float32)
    b = attn_head(a)
    print("b.shape:", b.shape)  # (batch, size, filter) => (batch, sizen, output_size)

