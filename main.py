import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="선수 유형 나누기",
    page_icon="⚽",
    layout="wide",
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "eafc25_top100.csv"
)


# =========================================================
# 능력치 이름
# =========================================================
ABILITY_COLUMNS = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움",
}


ABILITY_KEYS = list(ABILITY_COLUMNS.keys())

CIRCLE_LABELS = [
    "㉮",
    "㉯",
    "㉰",
    "㉱",
    "㉲",
    "㉳",
]


# =========================================================
# 포지션 분류
# =========================================================
POSITION_MAP = {
    # 공격수
    "ST": "공격수",
    "CF": "공격수",
    "LW": "공격수",
    "RW": "공격수",

    # 미드필더
    "LM": "미드필더",
    "RM": "미드필더",
    "CAM": "미드필더",
    "CM": "미드필더",
    "CDM": "미드필더",

    # 수비수
    "LWB": "수비수",
    "RWB": "수비수",
    "LB": "수비수",
    "RB": "수비수",
    "CB": "수비수",
    "GK": "수비수",
}


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8",
    )

    numeric_columns = [
        "overall",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "value_eur",
        "age",
        "height_cm",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["name_ko"] = df["name_ko"].fillna(
        df["name"]
    )

    df = df.dropna(
        subset=[
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physic",
            "overall",
        ]
    ).copy()

    return df


# =========================================================
# 앱 제목
# =========================================================
st.title("⚽ 선수 유형 나누기")

st.markdown(
    "EA FC 25 선수들의 능력치를 이용해 "
    "**비슷한 선수끼리 K-평균(K-Means)으로 묶어보는 앱**입니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================
try:

    df = load_data()

except Exception as e:

    st.error(
        "데이터를 불러오지 못했습니다."
    )

    st.write(e)

    st.stop()


# =========================================================
# 선수 유형 설정
# =========================================================
st.divider()

st.header("⚙️ 선수 유형 설정")


# ---------------------------------------------------------
# 사용할 능력치 선택
# ---------------------------------------------------------
selected_keys = st.multiselect(
    "묶는 데 사용할 능력치를 골라 주세요.",
    options=ABILITY_KEYS,
    default=ABILITY_KEYS,
    format_func=lambda x: ABILITY_COLUMNS[x],
)


if len(selected_keys) < 2:

    st.warning(
        "선수 유형을 나누려면 능력치를 2개 이상 선택해야 합니다."
    )

    st.stop()


# ---------------------------------------------------------
# 묶음 수
# ---------------------------------------------------------
cluster_count = st.slider(
    "묶음 수",
    min_value=2,
    max_value=6,
    value=3,
    step=1,
)


st.info(
    f"사용 능력치: "
    f"{', '.join(ABILITY_COLUMNS[x] for x in selected_keys)}"
    f"  |  현재 묶음 수: {cluster_count}개"
)


# =========================================================
# 표준화
# =========================================================
X = df[selected_keys].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# 묶음 수 결정하기
# =========================================================
st.divider()

st.header("📐 묶음 수 결정하기")

st.markdown(
    "현재 선택한 능력치를 표준화한 뒤, "
    "**묶음 수를 1~7까지 바꿔가며** "
    "K-Means를 다시 계산합니다."
)


# =========================================================
# K = 1 ~ 7 Inertia 계산
# =========================================================
inertia_values = []

k_values = list(range(1, 8))


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )


# =========================================================
# Inertia 데이터프레임
# =========================================================
inertia_table = pd.DataFrame(
    {
        "묶음 수": k_values,
        "중심에서 떨어진 거리의 제곱합": inertia_values,
    }
)


# 직전 값과의 감소량
decrease_values = [""]

for i in range(1, len(inertia_values)):

    decrease = (
        inertia_values[i - 1]
        - inertia_values[i]
    )

    decrease_values.append(decrease)


inertia_table["직전 값에서 감소한 정도"] = (
    decrease_values
)


# 숫자 표시용 복사본
inertia_display = inertia_table.copy()

inertia_display[
    "중심에서 떨어진 거리의 제곱합"
] = inertia_display[
    "중심에서 떨어진 거리의 제곱합"
].round(2)


for i in range(1, len(inertia_display)):

    inertia_display.loc[
        i,
        "직전 값에서 감소한 정도"
    ] = round(
        inertia_values[i - 1]
        - inertia_values[i],
        2,
    )


# =========================================================
# Inertia 꺾은선 그래프
# =========================================================
fig_inertia = px.line(
    inertia_table,
    x="묶음 수",
    y="중심에서 떨어진 거리의 제곱합",
    markers=True,
    title="묶음 수에 따른 중심에서 떨어진 거리의 제곱합",
    labels={
        "묶음 수": "묶음 수 (K)",
        "중심에서 떨어진 거리의 제곱합":
            "거리의 제곱합",
    },
)


# 현재 묶음 수에 세로선
fig_inertia.add_vline(
    x=cluster_count,
    line_width=3,
    line_dash="dash",
    annotation_text=f"현재 K = {cluster_count}",
    annotation_position="top",
)


fig_inertia.update_traces(
    marker=dict(
        size=9,
    )
)


fig_inertia.update_layout(
    height=550,
    hovermode="x unified",
)


st.plotly_chart(
    fig_inertia,
    use_container_width=True,
)


# =========================================================
# Inertia 표
# =========================================================
st.subheader(
    "📋 묶음 수별 거리의 제곱합과 감소량"
)

st.dataframe(
    inertia_display,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "거리의 제곱합이 작을수록 같은 묶음 안의 선수들이 "
    "각 묶음의 중심에 더 가깝게 모여 있다는 뜻입니다."
)


# =========================================================
# 실루엣 점수 계산
# =========================================================
st.divider()

st.subheader("🔎 실루엣 점수")

st.markdown(
    "실루엣 점수는 같은 묶음 안에서는 서로 가깝고, "
    "다른 묶음과는 얼마나 떨어져 있는지를 나타내는 지표입니다."
)


silhouette_k_values = list(range(2, 8))

silhouette_values = []


for k in silhouette_k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels,
    )

    silhouette_values.append(score)


# =========================================================
# 실루엣 점수 데이터프레임
# =========================================================
silhouette_table = pd.DataFrame(
    {
        "묶음 수": silhouette_k_values,
        "실루엣 점수": silhouette_values,
    }
)


silhouette_display = silhouette_table.copy()

silhouette_display[
    "실루엣 점수"
] = silhouette_display[
    "실루엣 점수"
].round(3)


# =========================================================
# 실루엣 점수 그래프
# =========================================================
fig_silhouette = px.line(
    silhouette_table,
    x="묶음 수",
    y="실루엣 점수",
    markers=True,
    title="묶음 수에 따른 실루엣 점수",
    labels={
        "묶음 수": "묶음 수 (K)",
        "실루엣 점수": "실루엣 점수",
    },
)


# 현재 묶음 수에 세로선
fig_silhouette.add_vline(
    x=cluster_count,
    line_width=3,
    line_dash="dash",
    annotation_text=f"현재 K = {cluster_count}",
    annotation_position="top",
)


fig_silhouette.update_traces(
    marker=dict(
        size=9,
    )
)


fig_silhouette.update_layout(
    height=550,
    hovermode="x unified",
)


st.plotly_chart(
    fig_silhouette,
    use_container_width=True,
)


# =========================================================
# 실루엣 점수 표
# =========================================================
st.subheader(
    "📋 묶음 수별 실루엣 점수"
)

st.dataframe(
    silhouette_display,
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "실루엣 점수는 높을수록 같은 묶음끼리는 가깝고 "
    "다른 묶음과는 구분되는 경향을 의미합니다."
)


# =========================================================
# 현재 선택한 묶음으로 최종 K-Means
# =========================================================
st.divider()

st.header("🎯 현재 묶음 수로 선수 유형 계산")


final_kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=20,
)


df["cluster_raw"] = final_kmeans.fit_predict(
    X_scaled
)


# =========================================================
# 슈팅 평균 계산
# =========================================================
shooting_means = (
    df.groupby("cluster_raw")["shooting"]
    .mean()
    .sort_values(
        ascending=False
    )
)


cluster_order = list(
    shooting_means.index
)


# =========================================================
# ㉮ ㉯ ㉰ 부여
# 슈팅 평균이 높은 순
# =========================================================
label_map = {}

for i, cluster_number in enumerate(
    cluster_order
):

    label_map[
        cluster_number
    ] = CIRCLE_LABELS[i]


df["묶음"] = df["cluster_raw"].map(
    label_map
)


# =========================================================
# 포지션 분류
# =========================================================
df["첫 포지션"] = (
    df["positions"]
    .fillna("")
    .astype(str)
    .str.split(r"[,/ ]+")
    .str[0]
)


df["포지션"] = (
    df["첫 포지션"]
    .map(POSITION_MAP)
    .fillna("기타")
)


# =========================================================
# 묶음별 인원 + 능력치 평균
# =========================================================
st.subheader(
    "📊 묶음별 인원과 능력치 평균"
)


summary = (
    df.groupby("묶음")
    .agg(
        인원=("name_ko", "size"),
        속도=("pace", "mean"),
        슈팅=("shooting", "mean"),
        패스=("passing", "mean"),
        드리블=("dribbling", "mean"),
        수비=("defending", "mean"),
        몸싸움=("physic", "mean"),
    )
    .reindex(
        CIRCLE_LABELS[
            :cluster_count
        ]
    )
)


summary_display = summary.copy()


for column in [
    "속도",
    "슈팅",
    "패스",
    "드리블",
    "수비",
    "몸싸움",
]:

    summary_display[column] = (
        summary_display[column]
        .round(1)
    )


st.dataframe(
    summary_display,
    use_container_width=True,
)


# =========================================================
# 묶음별 Overall TOP 5
# =========================================================
st.subheader(
    "🏆 묶음별 종합 능력치 TOP 5"
)


columns = st.columns(
    cluster_count
)


for i, label in enumerate(
    CIRCLE_LABELS[
        :cluster_count
    ]
):

    members = (
        df[
            df["묶음"] == label
        ]
        .sort_values(
            "overall",
            ascending=False,
        )
        .head(5)
    )


    with columns[i]:

        st.markdown(
            f"### {label}"
        )


        if len(members) == 0:

            st.write(
                "선수 없음"
            )

        else:

            for _, row in members.iterrows():

                st.write(
                    f"**{row['name_ko']}** "
                    f"· Overall {int(row['overall'])}"
                )


# =========================================================
# 2차원 산점도
# =========================================================
st.divider()

st.header("🔵 2차원 산점도")


axis_col1, axis_col2 = st.columns(2)


with axis_col1:

    x_key = st.selectbox(
        "가로축",
        options=selected_keys,
        index=0,
        format_func=lambda x:
            ABILITY_COLUMNS[x],
    )


with axis_col2:

    y_key = st.selectbox(
        "세로축",
        options=selected_keys,
        index=1,
        format_func=lambda x:
            ABILITY_COLUMNS[x],
    )


fig_2d = px.scatter(
    df,

    x=x_key,
    y=y_key,

    color="묶음",

    category_orders={
        "묶음":
            CIRCLE_LABELS[
                :cluster_count
            ]
    },

    hover_name="name_ko",

    hover_data={
        x_key: True,
        y_key: True,
        "묶음": True,
        "name_ko": False,
    },

    labels={
        x_key:
            ABILITY_COLUMNS[x_key],

        y_key:
            ABILITY_COLUMNS[y_key],

        "묶음":
            "선수 유형",
    },

    title=(
        f"{ABILITY_COLUMNS[x_key]} × "
        f"{ABILITY_COLUMNS[y_key]}"
    ),
)


fig_2d.update_traces(
    marker=dict(
        size=8,
    )
)


fig_2d.update_layout(
    height=650,
    legend_title_text="선수 유형",
)


st.plotly_chart(
    fig_2d,
    use_container_width=True,
)


# =========================================================
# 3차원 산점도
# =========================================================
st.divider()

st.header("🧊 3차원 산점도")


if len(selected_keys) < 3:

    st.info(
        "3차원 산점도를 보려면 "
        "묶는 데 사용할 능력치를 "
        "3개 이상 골라 주세요."
    )

else:

    axis_col1, axis_col2, axis_col3 = (
        st.columns(3)
    )


    with axis_col1:

        x_3d = st.selectbox(
            "X축",
            options=selected_keys,
            index=0,
            format_func=lambda x:
                ABILITY_COLUMNS[x],
            key="x_3d",
        )


    with axis_col2:

        y_3d = st.selectbox(
            "Y축",
            options=selected_keys,
            index=1,
            format_func=lambda x:
                ABILITY_COLUMNS[x],
            key="y_3d",
        )


    with axis_col3:

        z_3d = st.selectbox(
            "Z축",
            options=selected_keys,
            index=2,
            format_func=lambda x:
                ABILITY_COLUMNS[x],
            key="z_3d",
        )


    fig_3d = px.scatter_3d(
        df,

        x=x_3d,
        y=y_3d,
        z=z_3d,

        color="묶음",

        category_orders={
            "묶음":
                CIRCLE_LABELS[
                    :cluster_count
                ]
        },

        hover_name="name_ko",

        hover_data={
            x_3d: True,
            y_3d: True,
            z_3d: True,
            "묶음": True,
            "name_ko": False,
        },

        labels={
            x_3d:
                ABILITY_COLUMNS[x_3d],

            y_3d:
                ABILITY_COLUMNS[y_3d],

            z_3d:
                ABILITY_COLUMNS[z_3d],

            "묶음":
                "선수 유형",
        },

        title=(
            f"{ABILITY_COLUMNS[x_3d]} × "
            f"{ABILITY_COLUMNS[y_3d]} × "
            f"{ABILITY_COLUMNS[z_3d]}"
        ),
    )


    fig_3d.update_traces(
        marker=dict(
            size=3,
        )
    )


    fig_3d.update_layout(
        height=800,

        legend_title_text="선수 유형",

        margin=dict(
            l=0,
            r=0,
            t=70,
            b=0,
        ),
    )


    st.plotly_chart(
        fig_3d,
        use_container_width=True,
    )


# =========================================================
# 묶음 × 포지션 교차표
# =========================================================
st.divider()

st.header("⚽ 묶음 × 포지션")


cross_table = pd.crosstab(
    df["묶음"],
    df["포지션"],
)


cross_table = (
    cross_table
    .reindex(
        index=CIRCLE_LABELS[
            :cluster_count
        ],
        fill_value=0,
    )
    .reindex(
        columns=[
            "공격수",
            "미드필더",
            "수비수",
            "기타",
        ],
        fill_value=0,
    )
)


st.caption(
    "positions 열의 맨 앞 포지션을 기준으로 "
    "공격수·미드필더·수비수로 분류했습니다. "
    "포지션은 선수 유형을 묶을 때 사용하지 않았습니다."
)


st.dataframe(
    cross_table,
    use_container_width=True,
)


# =========================================================
# 데이터 정보
# =========================================================
with st.expander("ℹ️ 데이터 정보"):

    st.write(
        f"분석에 사용한 선수 수: **{len(df)}명**"
    )

    st.write(
        "클러스터링 방법: "
        "선택한 능력치를 표준화한 후 K-Means 적용"
    )

    st.write(
        "난수 고정값: random_state = 42"
    )

    st.write(
        "묶음 이름: 슈팅 평균이 높은 순으로 "
        "㉮ → ㉯ → ㉰ → ㉱ → ㉲ → ㉳"
    )

    st.write(
        "묶음 수 결정 그래프의 거리 제곱합은 "
        "K-Means의 inertia 값입니다."
    )
