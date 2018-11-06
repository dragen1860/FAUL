import  tensorflow as tf
from    mlp import single_layer_classifier
from    model import FAUL
import  math
from    tensorflow import flags

from    mnistFS import MnistFS
from    utils import MyInit, HookReport


FLAGS = tf.flags.FLAGS



class MetaAE(FAUL):



    def get_weights(self, c, factor, h_c, name):
        """

        :param c: channel of first conv output,
        :param factor: enlarge channel layer by layer, on factor
        :param h_c: channel of hidden
        :param name: scope name, we set reuse=tf.AUTO_REUSE
        :return:
        """

        self.c, self.factor, self.h_c = c, factor, h_c

        # save all variable
        vars = []
        # kernel size
        k = 3
        myinit = tf.contrib.layers.xavier_initializer()


        if name == 'encoder':
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                # layer1
                # [b, d, d, img_c] => [b, d, d, c]
                vars.append(tf.get_variable('w1', [1, 1, self.imgc, c], dtype=tf.float32, initializer=myinit))
                vars.append(tf.get_variable('b1', [c], dtype=tf.float32, initializer=tf.initializers.zeros()))

                # layer 2 ~ 7
                for idx in range(factor):
                    orig_idx = 0 if idx is 0 else (idx - 1)
                    # layer2, from [b, h>>idx, w>>idx, c>>orig_factor] => [h>>idx>>1, w>>idx>>1, c>>idx]
                    # intuitively, scale down size and scale up channels
                    vars.append(tf.get_variable('w'+str(2*idx+2), [3, 3, c<<orig_idx, c<<idx], dtype=tf.float32,
                                                initializer=myinit))
                    vars.append(tf.get_variable('b'+str(2*idx+2), [c<<idx], dtype=tf.float32,
                                                initializer=tf.initializers.zeros()))
                    vars.append(tf.get_variable('w'+str(2*idx+3), [3, 3, c<<idx, c<<idx], dtype=tf.float32,
                                                initializer=myinit))
                    vars.append(tf.get_variable('b'+str(2*idx+3), [c<<idx], dtype=tf.float32,
                                                initializer=tf.initializers.zeros()))
                    # x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')


                # layer counter, layer 8
                idx =  2 * factor + 2
                orig_factor = 0 if factor is 0 else factor - 1
                vars.append(tf.get_variable('w'+str(idx), [3, 3, c<<orig_factor, c<<orig_factor], dtype=tf.float32,
                                            initializer=myinit))
                vars.append(tf.get_variable('b'+str(idx), [c<<orig_factor], dtype=tf.float32,
                                            initializer=tf.initializers.zeros()))
                idx += 1

                # layer 9
                vars.append(tf.get_variable('w'+str(idx), [3, 3, c<<orig_factor, h_c], dtype=tf.float32,
                                            initializer=myinit))
                vars.append(tf.get_variable('b'+str(idx), [h_c], dtype=tf.float32,
                                            initializer=tf.initializers.zeros()))

                # record the number of encoder layer
                self.encoder_layer_num = idx
                print('encoder:', self.encoder_layer_num, self.encoder_var_num)

        elif name is 'decoder':
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                layer_counter = 1
                for idx in range(factor-1, -1, -1):
                    # layer1
                    orig_c = h_c if idx == (factor-1) else (c<<(idx+1)) # deal with the first layer
                    vars.append(tf.get_variable('w'+str(layer_counter), [3, 3, orig_c, c << idx], dtype=tf.float32,
                                                initializer=myinit))
                    vars.append(tf.get_variable('b'+str(layer_counter), [c << idx], dtype=tf.float32,
                                                initializer=tf.initializers.zeros()))
                    vars.append(tf.get_variable('w'+str(layer_counter+1), [3, 3, c << idx, c << idx], dtype=tf.float32,
                                                initializer=myinit))
                    vars.append(tf.get_variable('b'+str(layer_counter+1), [c << idx], dtype=tf.float32,
                                                initializer=tf.initializers.zeros()))
                    # tf.batch_to_space(tf.tile(x, [n ** 2, 1, 1, 1]), [[0, 0], [0, 0]], n)
                    layer_counter +=2

                layer_counter = 2 * factor + 1
                # layer7
                vars.append(tf.get_variable('w'+str(layer_counter), [3, 3, c, c], dtype=tf.float32,
                                            initializer=myinit))
                vars.append(tf.get_variable('b'+str(layer_counter), [c], dtype=tf.float32,
                                            initializer=tf.initializers.zeros()))
                layer_counter += 1

                # layer8
                vars.append(tf.get_variable('w'+str(layer_counter), [3, 3, c, self.imgc], dtype=tf.float32,
                                            initializer=myinit))
                vars.append(tf.get_variable('b'+str(layer_counter), [self.imgc], dtype=tf.float32,
                                            initializer=tf.initializers.zeros()))

                self.decoder_layer_num = layer_counter
                print('decoder:', self.decoder_layer_num, self.decoder_var_num)

        else:
            raise NotImplementedError

        for p in vars:
            print(p)
        return vars

    @property
    def encoder_var_num(self):
        return 2 * self.encoder_layer_num

    @property
    def decoder_var_num(self):
        return 2* self.decoder_layer_num

    def forward_encoder(self, x, vars):
        """

        :param x:
        :return:
        """
        idx = 0

        # layer1
        op = tf.nn.conv2d(x, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        idx += 2

        # layer2/3/4, factor=0,1,2
        for idx in range(2, 2 + self.factor * 4, 4): # step=4
            op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 1])
            op = tf.nn.leaky_relu(op)


            op = tf.nn.conv2d(op, vars[idx + 2], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 3])
            op = tf.nn.leaky_relu(op)

            op = tf.nn.avg_pool(op, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')


        # update variable pointer
        idx = 2 + self.factor * 4

        # layer5
        op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        op = tf.nn.leaky_relu(op)
        idx += 2

        # layer6
        op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        idx += 2

        # print(idx, len(vars))
        assert idx == len(vars)

        return op


    def forward_decoder(self, h, vars):
        """

        :param x:
        :return:
        """
        idx = 0

        op = h
        # layer1/2/3, factor=2,1,0
        for idx in range(0, self.factor * 4, 4): # step=4
            op = tf.nn.conv2d(op, vars[idx], strides=(1,1,1,1), padding='SAME')
            # print(vars[idx].name, vars[idx+1].name)
            op = tf.nn.bias_add(op, vars[idx + 1])
            op = tf.nn.leaky_relu(op)


            op = tf.nn.conv2d(op, vars[idx + 2], strides=(1,1,1,1), padding='SAME')
            op = tf.nn.bias_add(op, vars[idx + 3])
            op = tf.nn.leaky_relu(op)

            op = tf.batch_to_space(tf.tile(op, [2 ** 2, 1, 1, 1]), [[0, 0], [0, 0]], 2)

        # update variable pointer
        idx = self.factor * 4

        # layer4
        op = tf.nn.conv2d(op, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        op = tf.nn.leaky_relu(op)
        idx += 2
        # layer5
        op = tf.nn.conv2d(op, vars[idx + 0], strides=(1,1,1,1), padding='SAME')
        op = tf.nn.bias_add(op, vars[idx + 1])
        # op = tf.nn.sigmoid(op)
        idx += 2

        assert idx == len(vars)

        return op

    def forward_ae(self, x, vars):
        """

        :param x:
        :return:
        """
        # every layer contains 2 variable generally
        assert len(vars) == (self.encoder_var_num+self.decoder_var_num)

        vars_encoder = vars[:self.encoder_var_num]
        vars_decoder = vars[self.encoder_var_num:]

        op = self.forward_encoder(x, vars_encoder)
        op = self.forward_decoder(op, vars_decoder)

        return op

    def __init__(self, dbs, train_dir, **kwargs):
        super(MetaAE, self).__init__(dbs, train_dir, **kwargs)


        self.task_num = FLAGS.batchsz
        self.update_num = FLAGS.update_num
        self.update_lr = FLAGS.update_lr
        self.meta_lr = FLAGS.meta_lr


        # merge 2 variables list into a list
        # this is 1st time to get these variables, so it will create and return
        self.c, self.factor, self.h_c = kwargs['c'], kwargs['factor'], kwargs['h_c']
        self.h_d = self.imgsz >> self.factor
        self.vars = self.get_weights(self.c, self.factor, self.h_c, 'encoder') + \
                    self.get_weights(self.c, self.factor, self.h_c, 'decoder')

        print('tasks:', self.task_num, 'update_lr:', self.update_lr, 'update_num:',
              self.update_num, 'meta lr:', self.meta_lr)
        print('n_way:', self.n_way, 'k_spt:', self.k_spt, 'k_qry:', self.k_qry)
        print('h:', [self.h_d, self.h_d, self.h_c], 'conv ch:', self.c, 'factor:', self.factor)

        # we will use these ops to get test process
        self.encoder_ops, self.decoder_ops, self.ae_ops, self.classify_ops = [], [], [], []

        # [b, 32, 32, 1]
        self.train_spt_x = tf.placeholder(tf.float32,
                                     [self.task_num, self.n_way * self.k_spt, self.imgsz, self.imgsz, self.imgc],
                                     name='train_spt_x')
        self.train_qry_x = tf.placeholder(tf.float32,
                                     [self.task_num, self.n_way * self.k_qry, self.imgsz, self.imgsz, self.imgc],
                                     name='train_qry_x')

        # [b, 10]
        # trian_spt_y and train_qry_y will NOT be used since its unsupervised training
        # but we will use test_spt_y and test_qry_y to get performance benchmark.
        self.train_spt_y = tf.placeholder(tf.float32, [self.task_num, self.n_way * self.k_spt], name='train_spt_y')
        self.train_qry_y = tf.placeholder(tf.float32, [self.task_num, self.n_way * self.k_qry], name='train_qry_y')

        # [5, 32, 32, 1] [75, 32, 32, 1]
        self.test_spt_x = tf.placeholder(tf.float32, [self.n_way * self.k_spt, self.imgsz, self.imgsz, self.imgc],
                                    name='test_spt_x')
        self.test_qry_x = tf.placeholder(tf.float32, [self.n_way * self.k_qry, self.imgsz, self.imgsz, self.imgc],
                                    name='test_qry_x')
        # []
        self.test_spt_y = tf.placeholder(tf.float32, [self.n_way * self.k_spt], name='test_spt_y')
        self.test_qry_y = tf.placeholder(tf.float32, [self.n_way * self.k_qry], name='test_qry_y')

        # [b, 4, 4, 16]
        self.h = tf.placeholder(tf.float32, [None, self.h_d, self.h_d, self.h_c], name='h')

    def task_metalearn(self, task_input):
        """
        create single task op, we need call this func multiple times to create a bunches of ops
        for several tasks
        NOTICE: this function will use outer `vars`.
        """
        x_spt, x_qry = task_input
        print('x_spt:', x_spt.shape, 'x_qry:', x_qry.shape)

        preds_qry, losses_qry, accs_qry = [], [], []

        pred_spt = self.forward_ae(x_spt, self.vars)
        loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)

        grads = tf.gradients(loss_spt, self.vars)
        # if FLAGS.stop_grad:
        #     grads = [tf.stop_gradient(grad) for grad in grads]
        fast_weights = list(map(lambda x:x[0] - self.update_lr * x[1], zip(self.vars, grads)))
        pred_qry = self.forward_ae(x_qry, fast_weights)
        preds_qry.append(pred_qry)
        loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
        losses_qry.append(loss_qry)

        for _ in range(self.update_num - 1):
            pred_spt = self.forward_ae(x_spt, fast_weights)
            loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)
            grads = tf.gradients(loss_spt, fast_weights)
            # if FLAGS.stop_grad:
            #     grads = [tf.stop_gradient(grad) for grad in grads]
            fast_weights = list(map(lambda x: x[0] - self.update_lr * x[1], zip(fast_weights, grads)))
            pred_qry = self.forward_ae(x_qry, fast_weights)
            preds_qry.append(pred_qry)
            loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
            losses_qry.append(loss_qry)


        return [pred_spt, preds_qry, loss_spt, losses_qry]

    def record_test_ops(self, vars_k, k):
        """
        record intermediate test status for update_num pretrain
        :param vars_k:
        :param k: just for log_tensor
        :return:
        """
        # add 0 updated representation
        encoder_op = self.forward_encoder(self.test_spt_x, vars_k[:self.encoder_var_num])
        decoder_op = self.forward_decoder(self.h, vars_k[self.encoder_var_num:])  # reuse
        ae_op = self.forward_ae(self.test_spt_x, vars_k)
        self.encoder_ops.append(encoder_op)
        self.decoder_ops.append(decoder_op)
        self.ae_ops.append(ae_op)

        # this op only optimize classifier, hence stop_gradient after encoder_op
        # classify_op is not a single op, including prediction and loss

        classify_loss, classify_pred = single_layer_classifier(tf.stop_gradient(encoder_op),
                                                            self.test_spt_y-5, self.n_way,
                                                            scope='classifier_%d'%k, reuse=False)
        classify_train_op = tf.train.AdamOptimizer().minimize(classify_loss)
        # print(test_qry_y, classify_pred)
        _, classify_acc = tf.metrics.accuracy(self.test_spt_y-5, classify_pred)

        self.classify_ops.append([classify_train_op, classify_loss, classify_pred, classify_acc])

        # record classification loss on latent
        # utils.HookReport.log_tensor(tf.reduce_mean(classify_loss), 'test_classify_h_loss_update_%d'%k)

        return

    def model(self, h_c, c, factor, training=True):
        """
        :param h_c: latent channel
        :param c: basic channel number
        :param factor: channel factor
        :param training: train or test
        :return:
        """
        task_num, meta_lr, update_lr, update_num = self.task_num, self.meta_lr, self.update_lr, self.update_num
        n_way, k_spt, k_qry = self.n_way, self.k_spt, self.k_qry


        out_dtype = [tf.float32, [tf.float32] * update_num, tf.float32, [tf.float32] * update_num]
        # out_dtype.extend([tf.float32, [tf.float32]*update_num])
        self.pred_spt, self.preds_qry, self.loss_spt, self.losses_qry = tf.map_fn(self.task_metalearn,
                                                                        elems=[self.train_spt_x, self.train_qry_x],
                                                                        dtype=out_dtype, parallel_iterations=task_num)

        self.loss_spt = tf.reduce_sum(self.loss_spt) / tf.to_float(task_num)
        self.losses_qry = [tf.reduce_sum(self.losses_qry[j]) / tf.to_float(task_num) for j in range(update_num)]


        optimizer = tf.train.AdamOptimizer(meta_lr)
        gvs = optimizer.compute_gradients(self.losses_qry[update_num - 1])
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        meta_op = optimizer.apply_gradients(gvs, global_step=tf.train.get_global_step())


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.meta_op = tf.group([meta_op])

        for i in range(update_num):
            # print(losses_qry[i])
            HookReport.log_tensor(self.losses_qry[i], 'train_loss_qry%d' % i)
            # utils.HookReport.log_tensor(tf.sqrt(self.losses_qry[i]) * 127.5, 'rmse%d'%i)


        #=================================================================================
        self.test_losses_spt = []
        # record before pretrain, update_k = 0
        self.record_test_ops(self.vars, 0)

        # starting from parameters: vars!
        pred_x = self.forward_ae(self.test_spt_x, self.vars)
        loss = tf.losses.mean_squared_error(pred_x, self.test_spt_x)
        self.test_losses_spt.append(loss)
        grads = tf.gradients(loss, self.vars)
        fast_weights = list(map(lambda p:p[0] - update_lr * p[1], zip(self.vars, grads)))
        # record update_k=1 status
        self.record_test_ops(fast_weights, 1)

        for i in range(1, update_num):
            pred_x = self.forward_ae(self.test_spt_x, fast_weights)
            loss = tf.losses.mean_squared_error(pred_x, self.test_spt_x)
            self.test_losses_spt.append(loss)
            grads = tf.gradients(loss, fast_weights)
            fast_weights = list(map(lambda p:p[0] - update_lr * p[1], zip(fast_weights, grads)))
            # record update=i+1 status
            self.record_test_ops(fast_weights, i+1)

        # treat the last update step as pretrain_op
        self.pretrain_op = fast_weights


        # def gen_images():
        #     return self.make_sample_grid_and_save(ops, interpolation=16, height=16)
        #
        # recon, inter, slerp, samples = tf.py_func(gen_images, [], [tf.float32]*4)
        # tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        # tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        # tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        # tf.summary.image('samples', tf.expand_dims(samples, 0))





def main(argv):
    print(FLAGS.flag_values_dict())

    train_db = MnistFS('ae_data/mnist', mode='train')
    test_db = MnistFS('ae_data/mnist', mode='test')
    dbs = {
        'train_db': train_db,
        'test_db': test_db,
        'imgsz': 32,
        'imgc': 1,
        'name': 'mnist',
        'n_way': 5,
        'k_spt': 1,
        'k_qry': 15
    }

    factor = 1 # int(round(math.log(32 // FLAGS.h_d, 2)))
    model = MetaAE(dbs, FLAGS.train_dir,
                    h_c = FLAGS.h_c,
                    c = FLAGS.c,
                    factor = factor)
    model.train()

    #


if __name__ == '__main__':
    import  os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    flags.DEFINE_string('train_dir', './logs','Folder where to save training data.')
    flags.DEFINE_float('meta_lr', 0.001, 'meta Learning rate.')
    flags.DEFINE_float('update_lr', 0.05, 'update Learning rate.')
    flags.DEFINE_integer('update_num', 5, 'inner update steps')
    flags.DEFINE_integer('batchsz', 8, 'tasks number for meta-learning')
    flags.DEFINE_string('db_name', 'mnist32', 'Data to train on.')
    flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')
    flags.DEFINE_integer('c', 16, 'Depth of first for convolution.')
    flags.DEFINE_integer('h_c', 8, 'Latent depth = depth multiplied by latent_width ** 2.')
    flags.DEFINE_integer('h_d', 4, 'Width of the latent space.')
    tf.app.run(main)