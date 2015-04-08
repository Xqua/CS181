# Copy this to ipython !
# Copy the following text then go to ipython console
# Then type:
# %paste
# which is the magic command


def test1(s):
    for i in range(1000):
        ui = users[s[i][0]]
        ai = artists[s[i][1]]
        mat[ui][ai] = s[i][2]
    print "done"


def test2(s):
    for i in range(1000):
        ui = users_l.index(s[i][0])
        ai = artists_l.index(s[i][1])
        mat[ui][ai] = s[i][2]
    print "done"


def test3(df):
    for i in range(1000):
        u = df['user'][i]
        a = df['artist'][i]
        ui = users_l.index(u)
        ai = artists_l.index(a)
        mat[ui][ai] = df['plays'][i]


def test4(df):
    for i in range(1000):
        u = df['user'][i]
        a = df['artist'][i]
        ui = users[u]
        ai = artists[a]
        mat[ui][ai] = df['plays'][i]

df = pd.read_csv('train.csv')
ds = df.values

users_l = sorted(df['user'].unique())
N_users = len(users_l)
artists_l = sorted(df['artist'].unique())
N_artists = len(artists_l)
users = {}
artists = {}
for i in range(N_users):
    users[users_l[i]] = i
for i in range(N_artists):
    artists[artists_l[i]] = i

%timeit - n 1 - r 3 test1(ds)
%timeit - n 1 - r 3 test2(ds)
%timeit - n 1 - r 3 test3(df)
%timeit - n 1 - r 3 test4(df)
