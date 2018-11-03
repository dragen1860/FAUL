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
        myinit = MyInit(0.2) # tf.contrib.layers.xavier_initializer()


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

    def model(self, h_c, c, factor, training=True):
        """
        :param h_c: latent channel
        :param c: basic channel number
        :param factor: channel factor
        :param training: train or test
        :return:
        """
        # meta batch size
        task_num = FLAGS.batchsz
        update_num = FLAGS.update_num
        update_lr = FLAGS.update_lr
        meta_lr = FLAGS.meta_lr

        # get hidden features maps dim/height/width and channel number
        h_d = self.imgsz >> factor
        n_way, k_spt, k_qry = self.n_way, self.k_spt, self.k_qry

        print('tasks:', task_num, 'update_lr:', update_lr, 'update_num:', update_num, 'meta lr:', meta_lr)
        print('n_way:', n_way, 'k_spt:', k_spt, 'k_qry:', k_qry)
        print('h:', [h_d, h_d, h_c], 'conv ch:', c, 'factor:', factor)

        def task_metalearn(task_input):
            """
            create single task op, we need call this func multiple times to create a bunches of ops
            for several tasks
            NOTICE: this function will use outer `vars`.
            """
            x_spt, x_qry = task_input
            print('x_spt:', x_spt.shape, 'x_qry:', x_qry.shape)

            preds_qry, losses_qry, accs_qry = [], [], []

            pred_spt = self.forward_ae(x_spt, vars)
            loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)

            grads = tf.gradients(loss_spt, vars)
            # if FLAGS.stop_grad:
            #     grads = [tf.stop_gradient(grad) for grad in grads]
            fast_weights = list(map(lambda x:x[0] - update_lr * x[1], zip(vars, grads)))
            pred_qry = self.forward_ae(x_qry, fast_weights)
            preds_qry.append(pred_qry)
            loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
            losses_qry.append(loss_qry)

            for _ in range(update_num - 1):
                pred_spt = self.forward_ae(x_spt, fast_weights)
                loss_spt = tf.losses.mean_squared_error(labels=x_spt, predictions=pred_spt)
                grads = tf.gradients(loss_spt, fast_weights)
                # if FLAGS.stop_grad:
                #     grads = [tf.stop_gradient(grad) for grad in grads]
                fast_weights = list(map(lambda x: x[0] - update_lr * x[1], zip(fast_weights, grads)))
                pred_qry = self.forward_ae(x_qry, fast_weights)
                preds_qry.append(pred_qry)
                loss_qry = tf.losses.mean_squared_error(labels=x_qry, predictions=pred_qry)
                losses_qry.append(loss_qry)


            task_output = [pred_spt, preds_qry, loss_spt, losses_qry]
            return task_output



        # merge 2 variables list into a list
        # this is 1st time to get these variables, so it will create and return
        vars = self.get_weights(c, factor, h_c, 'encoder') + self.get_weights(c, factor, h_c, 'decoder')

        ######################################################
        if training:
            # [b, 32, 32, 1]
            train_spt_x = tf.placeholder(tf.float32, [task_num, n_way * k_spt, self.imgsz, self.imgsz, self.imgc],
                                         name='train_spt_x')
            train_qry_x = tf.placeholder(tf.float32, [task_num, n_way * k_qry, self.imgsz, self.imgsz, self.imgc],
                                         name='train_qry_x')
            # [b, 10]
            # trian_spt_y and train_qry_y will NOT be used since its unsupervised training
            # but we will use test_spt_y and test_qry_y to get performance benchmark.
            train_spt_y = tf.placeholder(tf.float32, [task_num, n_way * k_spt], name='train_spt_y')
            train_qry_y = tf.placeholder(tf.float32, [task_num, n_way * k_qry], name='train_qry_y')

            out_dtype = [tf.float32, [tf.float32] * update_num, tf.float32, [tf.float32] * update_num]
            # out_dtype.extend([tf.float32, [tf.float32]*update_num])
            pred_spt, preds_qry, loss_spt, losses_qry = \
                tf.map_fn(task_metalearn, elems=[train_spt_x, train_qry_x], dtype=out_dtype, parallel_iterations=task_num)

            self.loss_spt = tf.reduce_sum(loss_spt) / tf.to_float(task_num)
            self.losses_qry = [tf.reduce_sum(losses_qry[j]) / tf.to_float(task_num) for j in range(update_num)]
            self.pred_spt, self.preds_qry = pred_spt, preds_qry
            del pred_spt, preds_qry, loss_spt, losses_qry
            # self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            # self.total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(loss_spt)

            optimizer = tf.train.AdamOptimizer(meta_lr)
            gvs = optimizer.compute_gradients(self.losses_qry[update_num - 1])
            # gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
            meta_op = optimizer.apply_gradients(gvs, global_step=tf.train.get_global_step())

            for i in range(update_num):
                # print(losses_qry[i])
                HookReport.log_tensor(self.losses_qry[i], 'train_loss_qry%d' % i)
                # utils.HookReport.log_tensor(tf.sqrt(self.losses_qry[i]) * 127.5, 'rmse%d'%i)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                meta_op = tf.group([meta_op])

        #=========================================================
        # [5, 32, 32, 1] [75, 32, 32, 1]
        test_spt_x = tf.placeholder(tf.float32, [n_way * k_spt, self.imgsz, self.imgsz, self.imgc],
                                    name='test_spt_x')
        test_qry_x = tf.placeholder(tf.float32, [n_way * k_qry, self.imgsz, self.imgsz, self.imgc],
                                    name='test_qry_x')
        # []
        test_spt_y = tf.placeholder(tf.float32, [n_way * k_spt], name='test_spt_y')
        test_qry_y = tf.placeholder(tf.float32, [n_way * k_qry], name='test_qry_y')

        # [b, 4, 4, 16]
        h = tf.placeholder(tf.float32, [None, h_d, h_d, h_c], name='h')


        # we will use these ops to get test process
        encoder_ops, decoder_ops, ae_ops, classify_accs = [], [], [], []

        def record_test_ops(vars_k, k):
            """
            record intermediate test status for update_num pretrain
            :param vars_k:
            :param k: just for log_tensor
            :return:
            """
            # add 0 updated representation
            encoder_op = self.forward_encoder(test_qry_x, vars_k[:self.encoder_var_num])
            decoder_op = self.forward_decoder(h, vars_k[self.encoder_var_num:])  # reuse
            ae_op = self.forward_ae(test_qry_x, vars_k)
            encoder_ops.append(encoder_op)
            decoder_ops.append(decoder_op)
            ae_ops.append(ae_op)

            # this op only optimize classifier, hence stop_gradient after encoder_op
            # classify_op is not a single op, including prediction and loss
            classify_loss, _ = single_layer_classifier(tf.stop_gradient(encoder_op),
                                                                test_qry_y, self.n_way,
                                                                scope='classifier_%d'%k, reuse=False)
            def train_mlp():
                with tf.Session() as sess:
                    op = tf.train.AdamOptimizer().minimize(classify_loss)
                    for _ in range(100):
                        sess.run(op)
                    classify_pred = single_layer_classifier(tf.stop_gradient(encoder_op),
                                                                test_qry_y, self.n_way,
                                                                scope='classifier_%d'%k, reuse=True)
                    classify_pred = sess.run(classify_pred)
                return classify_pred

            classify_pred = tf.py_func(train_mlp, None, [tf.int32])
            acc = tf.metrics.accuracy(test_qry_y, classify_pred)
            classify_accs.append(acc)
            # record classification loss on latent
            # utils.HookReport.log_tensor(tf.reduce_mean(classify_loss), 'test_classify_h_loss_update_%d'%k)

            return


        # record before pretrain, update_k = 0
        record_test_ops(vars, 0)

        # starting from parameters: vars!
        pred_x = self.forward_ae(test_spt_x, vars)
        loss = tf.losses.mean_squared_error(pred_x, test_spt_x)
        grads = tf.gradients(loss, vars)
        fast_weights = list(map(lambda p:p[0] - update_lr * p[1], zip(vars, grads)))
        # record update_k=1 status
        record_test_ops(fast_weights, 1)

        for i in range(1, update_num):
            pred_x = self.forward_ae(test_spt_x, fast_weights)
            loss = tf.losses.mean_squared_error(pred_x, test_spt_x)
            grads = tf.gradients(loss, fast_weights)
            fast_weights = list(map(lambda p:p[0] - update_lr * p[1], zip(fast_weights, grads)))
            # record update=i+1 status
            record_test_ops(fast_weights, i+1)

        # treat the last update step as pretrain_op
        pretrain_op = fast_weights
        ######################################################


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            pretrain_op = tf.group([pretrain_op])

        ops = {
            'meta_op': meta_op if training else None,
            'train_spt_x': train_spt_x if training else None,
            'train_spt_y': train_spt_y if training else None,
            'train_qry_x': train_qry_x if training else None,
            'train_qry_y': train_qry_y if training else None,

            'pretrain_op' : pretrain_op,
            'classify_preds': classify_preds, # len=update_num+1, array of (op.output, op.loss)
            'encoder_ops' : encoder_ops, # len=update_num+1,
            'decoder_ops' : decoder_ops, # len=update_num+1,
            'ae_ops'      : ae_ops, # len=update_num+1

            'test_spt_x': test_spt_x,
            'test_spt_y': test_spt_y,
            'test_qry_x': test_qry_x,
            'test_qry_y': test_qry_y
        }


        # def gen_images():
        #     return self.make_sample_grid_and_save(ops, interpolation=16, height=16)
        #
        # recon, inter, slerp, samples = tf.py_func(gen_images, [], [tf.float32]*4)
        # tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        # tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        # tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        # tf.summary.image('samples', tf.expand_dims(samples, 0))

        return ops




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