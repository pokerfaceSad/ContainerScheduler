from config import *
from environment import *
from service_batch_generator import *
from agent import *
from geneticAlgorithm.solver import *

""" Configuration """
config, _ = get_config()
config.batch_size = 10
isRender = True
isCmpGA = False
""" Environment """
env = Environment(config.num_bins, config.num_slots, config.num_descriptors)

""" Batch of Services """
services = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, config.num_descriptors)

""" Agent """
state_size_sequence = config.max_length
state_size_embeddings = config.num_descriptors  # OH Vector embedding
action_size = config.num_bins
agent = Agent(state_size_embeddings, state_size_sequence, action_size, config.batch_size, config.learning_rate,
              config.hidden_dim, config.num_stacks)

""" Configure Saver to save & restore model variables """
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # total_parameters = 0
    # for variable in tf.trainable_variables():
    #     # shape is an array of tf.Dimension
    #     shape = variable.get_shape()
    #     variable_parameters = 1
    #     for dim in shape:
    #         variable_parameters *= dim.value
    #     print('Shape: ', shape, 'Variables: ', variable_parameters)
    #     total_parameters += variable_parameters
    # print('Total_parameters: ', total_parameters)

    if config.load_model:
        saver.restore(sess, "save/tf_binpacking.ckpt")
        # saver.restore(sess, "save/tmp.ckpt-41400")
        print("Model restored.")

    # New batch of states
    services.getNewState()

    # Vector embedding
    input_state = vector_embedding(services)

    # Compute placement
    feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
    # prob_positions = sess.run(agent.ptr.prob_positions, feed_dict=feed)
    best_positions = sess.run(agent.ptr.best_positions, feed_dict=feed)
    # prob_reward = np.zeros(config.batch_size)
    best_reward = np.zeros(config.batch_size)

    # Compute environment
    for batch in range(config.batch_size):
        # env.clear()
        # env.step(prob_positions[batch], services.state[batch], services.serviceLength[batch])
        # prob_reward[batch] = env.reward
        env.clear()
        env.step(best_positions[batch], services.state[batch], services.serviceLength[batch])
        best_reward[batch] = env.reward


        # Render some batch services
        # if batch % max(1, int(config.batch_size / 5)) == 0:
        if isRender:
            print("\n Rendering batch ", batch, "...")
            env.render(batch)

        if isCmpGA:
            env.clear()
            chromo = solve(serviceChain=services.state[batch], serviceLength=services.serviceLength[batch], binCapacity=config.num_slots)
            position = chrom2position(chromo)
            env.step(position, services.state[batch], services.serviceLength[batch])
            env.render()

    print(best_reward)
    # print(prob_reward)