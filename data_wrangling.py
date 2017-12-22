import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"])

attack_dict = {'normal': 'normal','back': 'DoS','land': 'DoS','neptune': 'DoS','pod': 'DoS',
               'smurf': 'DoS','teardrop': 'DoS','mailbomb': 'DoS','apache2': 'DoS','processtable': 'DoS',
               'udpstorm': 'DoS',

               'ipsweep': 'Probe','nmap': 'Probe','portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe',
               'saint': 'Probe',

               'ftp_write': 'R2L','guess_passwd': 'R2L','imap': 'R2L','multihop': 'R2L', 'phf': 'R2L','spy': 'R2L',
                'warezclient': 'R2L','warezmaster': 'R2L','sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
               'snmpguess': 'R2L','xlock': 'R2L','xsnoop': 'R2L','worm': 'R2L',

               'buffer_overflow': 'U2R','loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R','httptunnel': 'U2R',
               'ps': 'U2R','sqlattack': 'U2R','xterm': 'U2R'
}

FILE_PATH = './kdd/'
dummy_list = ['protocol_type','service','flag']
log_scaling_list = ['duration','src_bytes','dst_bytes']
binary_dict = {'DoS' : 1,  'normal':0, 'Probe':1 ,'R2L':1, 'U2R':1}


def _data_load():
    '''
    '''
    cnt = 0
    file_list = ['KDDTrain+','KDDTest+', 'KDDTest-21']
    for name in file_list:
        if cnt < 1:
            result = pd.read_csv(FILE_PATH+name+'.txt', header=None)
            cnt += 1
        else:
            result.append(pd.read_csv(FILE_PATH+name+'.txt', header=None))
    result = result.iloc[:, 0:42]  # 끝행 지우기
    return result

def _concat_dummy(df, var):

    for name in var:
        df = pd.concat([df,pd.get_dummies(df[name])], axis=1)
    return df.drop(var , axis=1)


def _min_max_scalier(df, binary_classify=True):

    df = df.drop(['labels'],axis=1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def one_hot_encoder(df, var):
    df_y = df['labels'].map(var)         #Normal : 0
    y_ = np.array(pd.get_dummies(df_y))  #원-핫 형태로 변형

    return y_


df = _data_load()  # 데이터 로드
df.columns = col_names  # 헤더값 정의
df['labels'] = df['labels'].map(attack_dict)  # 위협 별 종류 맵핑  ex) ps : 'U2R'
df = _concat_dummy(df, dummy_list) # 더미변수처리
df[log_scaling_list].apply(lambda x:np.log(x+0.1)) #로그스케일 처리
new_df = _min_max_scalier(df)
y_ = one_hot_encoder(df, binary_dict)


X_train, X_test, y_train, y_test = train_test_split(new_df, y_, test_size = 0.33, random_state = 42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
