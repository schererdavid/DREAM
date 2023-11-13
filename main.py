
import argparse
import os
import time

def str_to_list(string_representation):
    import ast
    try:
        my_list = ast.literal_eval(string_representation)
        if not isinstance(my_list, list):
            print("The provided string does not represent a list.")
    except (SyntaxError, ValueError):
        print("Invalid string representation.")
    return my_list

def main():
    parser = argparse.ArgumentParser(description='Predicts stop of antibiotics')
    parser.add_argument('-t', '--task', help='task which should be performed')
    parser.add_argument('-d', '--database', help='database which should be used for the task')
    parser.add_argument('-ab', '--inc_ab', help='wheter or not to include antibiotics as features')
    parser.add_argument('-m', '--microbiologyneeded', help='if a microbiology result is needed or not')
    parser.add_argument('-s', '--seed', help='seed used for random states')

    # traditional arguments
    parser.add_argument('-l', '--lookbacks', help='list of maximum lookbacks at prediction time point, e.g. [2]')
    parser.add_argument('-tp', '--timepoints', help='list of prediction time points, e.g. [0,1] or the string random')
    parser.add_argument('-ns', '--numberofsamples', help='number of samples taken if timepoint is random')
    parser.add_argument('-ste', '--sampletest', help='if test set should be resampled')
    parser.add_argument('-str', '--sampletrain', help='if train set should be resampled')

    # next day arguments
    parser.add_argument('-f', '--fast', help='if fast mode is activated for lstm training')
    parser.add_argument('-nl', '--num_lin', help='number of linear layers')
    parser.add_argument('-nsl', '--num_stacked_lstm', help='number of stacked lstms')
    parser.add_argument('-hd', '--hidden_dim', help='hidden dimension of linear layers and lstm')
    parser.add_argument('-dr', '--dropout', help='dropout in next day network')
    parser.add_argument('-la', '--lamb', help='lambda for the custom loss function, 0 = only next day loss')
    parser.add_argument('-it', '--is_tuned', help='whether or not we use the tuned label in the next day model training')
    parser.add_argument('-lr', '--learning_rate', help='initial learning rate of next day model')
    parser.add_argument('-bs', '--batch_size', help='batch size for the next day model')
    parser.add_argument('-re', '--use_relus', help='if relus should be used')
    parser.add_argument('-bn', '--use_batchnormalization', help='if batchnormalization should be used')
    parser.add_argument('-ll', '--lookback_lstm', help='max lookback of lstm')
    parser.add_argument('-al', '--aggregation_lstm', help='number of hours per aggregation bin for lstm')
    parser.add_argument('-c', '--censored', help='wheter or not to include censored patients')

    args = parser.parse_args()


    task = args.task
    database = args.database

    if args.censored != None:
        censored = eval(args.censored)
    else:
        censored = None

    if args.inc_ab != None:
        inc_ab = eval(args.inc_ab)
    else:
        inc_ab = None

    if args.microbiologyneeded != None:
        has_microbiology = eval(args.microbiologyneeded)
    else:
        has_microbiology = None
    
    if args.fast != None:
        fast = eval(args.fast)
    else:
        fast = None

    if args.lookbacks != None:
        lookbacks = str_to_list(args.lookbacks)
    else:
        lookbacks = None
    
    if args.timepoints != None:
        if args.timepoints == 'random':
            timepoints = args.timepoints
        else:
            timepoints = str_to_list(args.timepoints)
    else:
        timepoints = None

    if args.sampletest != None:
        sampletest = args.sampletest
    else:
        sampletest = None
    
    if args.sampletrain != None:
        sampletrain = args.sampletrain
    else:
        sampletrain = None

    if args.numberofsamples != None:
        numberofsamples = int(args.numberofsamples)
    else:
        numberofsamples = None

    if args.seed != None:
        seed = int(args.seed)
    else:
        seed = None

    if args.num_lin != None:
        num_lin = int(args.num_lin)
    else:
        num_lin = None

    if args.dropout != None:
        dropout = float(args.dropout)
    else:
        dropout = None

    if args.lamb != None:
        lamb = float(args.lamb)
    else:
        lamb = None

    if args.learning_rate != None:
        lr = float(args.learning_rate)
    else:
        lr = None

    if args.batch_size != None:
        bs = int(args.batch_size)
    else:
        bs = None
    
    if args.hidden_dim != None:
        hidden_dim = int(args.hidden_dim)
    else:
        hidden_dim = None

    if args.num_stacked_lstm != None:
        num_stacked_lstm = int(args.num_stacked_lstm)
    else:
        num_stacked_lstm = None

    if args.is_tuned != None:
        is_tuned = eval(args.is_tuned)
    else:
        is_tuned = None

    if args.use_relus != None:
        use_relus = eval(args.use_relus)
    else:
        use_relus = None

    if args.use_batchnormalization != None:
        use_batchnormalization = eval(args.use_batchnormalization)
    else:
        use_batchnormalization = None

    if args.lookback_lstm != None:
        lookback_lstm = int(args.lookback_lstm)
    else:
        lookback_lstm = None

    if args.aggregation_lstm != None:
        aggregation_lstm = int(args.aggregation_lstm)
    else:
        aggregation_lstm = None
    
    print("======= Comandline Parameters ========")
    print("-------         General       --------")
    print('task:', task)
    print('database:', database)
    print('inc_ab:', inc_ab)
    print('has_microbiology:', has_microbiology)
    print('seed:', seed)
    print("-------     Traditional       --------")
    print('lookbacks:', lookbacks)
    print('timepoints:', timepoints)
    print('sampletest:', sampletest)
    print('sampletrain:', sampletrain)
    print('numberofsamples:', numberofsamples)
    print("-------       Next Day        --------")
    print('fast:', fast)
    print('num_lin:', num_lin)
    print('dropout:', dropout)
    print('lamb:', lamb)
    print('is_tuned:', is_tuned)
    print('batch_size:', bs)
    print('learning_rate:', lr)
    print('hidden_dim:', hidden_dim)
    print('num_stacked_lstm:', num_stacked_lstm)
    print('use_relus:', use_relus)
    print('use_batchnormalization:', use_batchnormalization)
    print('lookback_lstm:', lookback_lstm)
    print('aggregation_lstm:', aggregation_lstm)
    print("======================================")
    
    start_time = time.time()    

    if database not in ["mimic", "eicu", 'pic']:
        raise Exception("database currently not supported")

    # python3 main.py -t load -d mimic
    if task == "load" or task == "prepare":
        print("load concepts...")
        from concept_module.concept_loader import load_concepts
        load_concepts(database)

    
    # python3 main.py -t extract -d mimic
    if task == "extract" or task == "prepare":
        print("extract features...")
        from feature_module.feature_extractor import extract_features
        extract_features(database)

    # python3 main.py -t aggregate -d mimic    
    if task == "aggregate" or task == "prepare":
        print("aggregate to 5min intervals...")
        from feature_module.feature_aggregator import aggregate_to_5min
        aggregate_to_5min(database, method='median')


    #  python3 main.py -t episodes -d mimic -m True   
    if task == "episodes":
        print("episodes are constructed...")
        from episode_module.episode_constructor import construct_episodes
        construct_episodes(database, only_with_microbiology_res=has_microbiology)


    # python3 main.py -t split_test -d mimic    
    if task == "split_test" or task == "input":
        print("split into train test set...")
        from episode_module.episode_constructor import split_data_by_ids
        path = 'data/episodes/'+database+'/microbiology_res_'+str(has_microbiology)
        split_data_by_ids(sourname=path+'/all_episodes.parquet',
                            destination1=path+'/seed_'+str(seed)+'/train_data.parquet',
                            destination2=path+'/seed_'+str(seed)+'/test_data.parquet',
                            path=path,
                            test_size=0.15, 
                            random_state=seed)


    # python3 main.py -t split_validation -d mimic    
    # only for lstm approach, traditional uses cross validation    
    if task == "split_validation" or task == "input":
        print("split into train validation set...")
        from episode_module.episode_constructor import split_data_by_ids
        path = 'data/episodes/'+database+'/microbiology_res_'+str(has_microbiology)
        split_data_by_ids(sourname=path+"/seed_"+str(seed)+'/train_data.parquet', 
                            destination1=path+'/seed_'+str(seed)+'/train_lstm_data.parquet',
                            destination2=path+'/seed_'+str(seed)+'/validation_lstm_data.parquet',
                            path=path,
                            test_size=0.15, 
                            random_state=seed)


    # python3 main.py -t construct_traditional -d mimic -l [2] -tp random -ns 1 -s 42 -ab True -m True
    if task == "construct_traditional" or task == "input":
        print("construct_traditional")
        from dataset_module.dataset import construct_input_traditional
        construct_input_traditional(database=database, 
                                    lookbacks=lookbacks, 
                                    prediction_time_points=timepoints, 
                                    numberofsamples=numberofsamples, 
                                    seed=seed, 
                                    inc_ab=inc_ab,
                                    has_microbiology=has_microbiology)


    # python3 main.py -t construct_lstm -d mimic    
    if task == "construct_lstm":
        print("construct_lstm")
        from dataset_module.dataset import construct_X_lstm
        construct_X_lstm(database=database, 
                         lookback=lookback_lstm, 
                         aggregated_hours=aggregation_lstm, 
                         seed=seed, 
                         inc_ab=inc_ab,
                         has_microbiology=has_microbiology)


    # python3 main.py -d mimic -t train_traditional -ab False -m True -s 42 -l [2] -tp random -ns 1
    if task == "train_traditional" or task == "traditional_train_test_evaluate":
        print("train_traditional")
        from train_module.trainer import run_train_loop
        run_train_loop(database=database, 
                        lookbacks=lookbacks, 
                        prediction_time_points=timepoints, 
                        numberofsamples=numberofsamples, 
                        sample_train=sampletrain, 
                        sample_test=sampletest,
                        seed=seed, 
                        inc_ab=inc_ab,
                        has_microbiology=has_microbiology)

        
    if task == "train_lstm" or task == "lstm_train_test_evaluate":
        print("train_lstm")
        from train_module.lstm_trainer import train_lstm
        train_lstm(database=database, 
                    fast=fast,
                    seed=seed, 
                    inc_ab=inc_ab,
                    has_microbiology=has_microbiology,
                    use_censored=censored,
                    lookback=lookback_lstm, 
                    aggregated_hours=aggregation_lstm, 
                    num_lin_layers=num_lin,
                    dropout_prob=dropout,
                    lamb=lamb,
                    is_tuned = is_tuned, 
                    lr = lr,
                    bs = bs,
                    num_stacked_lstm = num_stacked_lstm,
                    hidden_dim = hidden_dim,
                    use_relus = use_relus, 
                    use_batchnormalization = use_batchnormalization)

        
    # python3 main.py -t test_traditional -d mimic -l [2] -c False -tp random -ns 1
    if task == "test_traditional" or task == "traditional_train_test_evaluate":
        print("test_traditional")
        from test_module.tester import run_test_loop
        run_test_loop(database=database, 
                        lookbacks=lookbacks, 
                        prediction_time_points=timepoints, 
                        numberofsamples=numberofsamples, 
                        sample_train=sampletrain, 
                        sample_test=sampletest, 
                        seed=seed,
                        inc_ab=inc_ab,
                        has_microbiology=has_microbiology)

        
    if task == "test_lstm" or task == "lstm_train_test_evaluate":
        print("test_lstm")
        from test_module.tester import test_lstm
        test_lstm(database = database, 
                    seed = seed,
                    fast = fast,  
                    inc_ab=inc_ab,
                    has_microbiology=has_microbiology,
                    use_censored=censored,
                    lookback=lookback_lstm, 
                    aggregated_hours=aggregation_lstm, 
                    num_lin_layers=num_lin,
                    dropout_prob=dropout,
                    lamb=lamb,
                    is_tuned = is_tuned, 
                    lr = lr,
                    bs = bs,
                    num_stacked_lstm = num_stacked_lstm,
                    hidden_dim = hidden_dim,
                    use_relus = use_relus, 
                    use_batchnormalization = use_batchnormalization)

        
    # python3 main.py -t evaluate_traditional -d mimic -l [2] -c False -tp random -ns 1
    if task == "evaluate_traditional" or task == "traditional_train_test_evaluate":
        print("evaluate_traditional")
        from test_module.tester import run_evaluation
        run_evaluation(database=database, 
                        lookbacks=lookbacks, 
                        prediction_time_points=timepoints, 
                        numberofsamples=numberofsamples, 
                        sample_train=sampletrain, 
                        sample_test=sampletest,
                        seed=seed, 
                        inc_ab=inc_ab,
                        has_microbiology=has_microbiology)


    if task == "evaluate_lstm" or task == "lstm_train_test_evaluate":
        print("evaluate_lstm")
        from test_module.tester import evaluate_lstm
        evaluate_lstm(database=database,
                        seed=seed, 
                        inc_ab=inc_ab,
                        has_microbiology=has_microbiology,
                        use_censored=censored,
                        lookback=lookback_lstm, 
                        aggregated_hours=aggregation_lstm, 
                        num_lin_layers=num_lin,
                        dropout_prob=dropout,
                        lamb=lamb,
                        is_tuned = is_tuned, 
                        lr = lr,
                        bs = bs,
                        num_stacked_lstm = num_stacked_lstm,
                        hidden_dim = hidden_dim,
                        use_relus = use_relus, 
                        use_batchnormalization = use_batchnormalization)

        
    if task == "evaluate_combined":
        print("evaluate combined")
        from test_module.tester import compare_models
        compare_models()

    if task == "independent":
        print("independent")
        from test_module.tester import independent
        independent()    
    
    end_time = time.time()
    runtime = end_time - start_time
    print("Runtime: ", str(runtime), " seconds, ", str(runtime / 60.0)," minutes")

    

    

if __name__ == "__main__":
    main()