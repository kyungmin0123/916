import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="선수 유형 나누기",
    page_icon="⚽",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/eafc25_top100.csv"

ABILITY_COLUMNS = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움",
}

CIRCLE_LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]

POSITION_MAP = {
    "ST": "공격수",
    "CF": "공격수",
    "LW": "공격수",
    "RW": "공격수",

    "LM": "미드필더",
    "RM": "미드필더",
    "CAM": "미드필더",
    "CM": "미드필더",
    "CDM": "미드필더",

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
        encoding="utf-8"
    )

    # 숫자형 변환
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

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["name_ko"] = df["name_ko"].fillna(df["name"])

    # 필요한 능력치가 없는 행 제거
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
# 제목
# =========================================================
st.title("⚽ 선수 유형 나누기")

st.markdown(
    "EA FC 25 선수들의 능력치를 이용해 "
    "**비슷한 선수끼리 K-평균(K-Means)으로 묶어보는 앱**입니다."
)


# =========================================================
# 데이터 로드
# =========================================================
try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(e)
    st.stop()


# =========================================================
# 능력치 선택
# =========================================================
st.divider()

st.header("⚙️ 선수 유형 설정")

ability_keys = list(ABILITY_COLUMNS.keys())

selected_keys = st.multiselect(
    "묶는 데 사용할 능력치를 골라 주세요.",
    options=ability_keys,
    default=ability_keys,
    format_func=lambda x: ABILITY_COLUMNS[x],
)

if len(selected_keys) < 2:
    st.warning(
        "선수 유형을 나누려면 능력치를 2개 이상 선택해야 합니다."
    )
    st.stop()


# =========================================================
# 묶음 수
# =========================================================
cluster_count = st.slider(
    "묶음 수",
    min_value=2,
    max_value=6,
    value=3,
    step=1,
)

st.info(
    f"선택한 능력치: "
    f"{', '.join(ABILITY_COLUMNS[x] for x in selected_keys)}  |  "
    f"묶음 수: {cluster_count}개"
)


# =========================================================
# K-Means
# =========================================================
X = df[selected_keys].copy()

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=20,
)

df["cluster_raw"] = kmeans.fit_predict(X_scaled)


# =========================================================
# 슈팅 평균이 높은 순으로 ㉮ ㉯ ㉰ 부여
# =========================================================
shooting_means = (
    df.groupby("cluster_raw")["shooting"]
    .mean()
    .sort_values(ascending=False)
)

cluster_order = list(shooting_means.index)

label_map = {}

for i, cluster_number in enumerate(cluster_order):
    label_map[cluster_number] = CIRCLE_LABELS[i]

df["묶음"] = df["cluster_raw"].map(label_map)


# =========================================================
# 포지션 분류
# positions의 맨 앞 포지션만 사용
# 클러스터링에는 사용하지 않음
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
# 결과
# =========================================================
st.divider()

st.header("📊 선수 유형 결과")


# =========================================================
# 묶음별 평균 능력치
# =========================================================
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
        CIRCLE_LABELS[:cluster_count]
    )
)

summary_display = summary.copy()

for col in [
    "속도",
    "슈팅",
    "패스",
    "드리블",
    "수비",
    "몸싸움",
]:
    summary_display[col] = summary_display[col].round(1)

st.subheader("📋 묶음별 인원과 능력치 평균")

st.dataframe(
    summary_display,
    use_container_width=True,
)


# =========================================================
# 묶음별 Overall TOP 5
# =========================================================
st.subheader("🏆 묶음별 종합 능력치 TOP 5")

columns = st.columns(cluster_count)

for i, label in enumerate(
    CIRCLE_LABELS[:cluster_count]
):

    members = (
        df[df["묶음"] == label]
        .sort_values(
            "overall",
            ascending=False
        )
        .head(5)
    )

    with columns[i]:

        st.markdown(f"### {label}")

        if len(members) == 0:

            st.write("선수 없음")

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

col1, col2 = st.columns(2)

with col1:

    x_key = st.selectbox(
        "가로축",
        options=selected_keys,
        index=0,
        format_func=lambda x: ABILITY_COLUMNS[x],
    )

with col2:

    y_key = st.selectbox(
        "세로축",
        options=selected_keys,
        index=1,
        format_func=lambda x: ABILITY_COLUMNS[x],
    )


fig_2d = px.scatter(
    df,
    x=x_key,
    y=y_key,
    color="묶음",

    category_orders={
        "묶음": CIRCLE_LABELS[:cluster_count]
    },

    hover_name="name_ko",

    hover_data={
        x_key: True,
        y_key: True,
        "묶음": True,
        "name_ko": False,
    },

    labels={
        x_key: ABILITY_COLUMNS[x_key],
        y_key: ABILITY_COLUMNS[y_key],
        "묶음": "선수 유형",
    },

    title=(
        f"{ABILITY_COLUMNS[x_key]} × "
        f"{ABILITY_COLUMNS[y_key]}"
    ),
)

fig_2d.update_traces(
    marker=dict(
        size=8
    )
)

fig_2d.update_layout(
    height=650,
    legend_title_text="선수 유형",
)

st.plotly_chart(
    fig_2d,
    use_container_width=True
)


# =========================================================
# 3차원 산점도
# =========================================================
st.divider()

st.header("🧊 3차원 산점도")

if len(selected_keys) < 3:

    st.info(
        "3차원 산점도를 보려면 "
        "묶는 데 사용할 능력치를 3개 이상 골라 주세요."
    )

else:

    col1, col2, col3 = st.columns(3)

    with col1:

        x_3d = st.selectbox(
            "X축",
            options=selected_keys,
            index=0,
            format_func=lambda x: ABILITY_COLUMNS[x],
            key="x_3d",
        )

    with col2:

        y_3d = st.selectbox(
            "Y축",
            options=selected_keys,
            index=1,
            format_func=lambda x: ABILITY_COLUMNS[x],
            key="y_3d",
        )

    with col3:

        z_3d = st.selectbox(
            "Z축",
            options=selected_keys,
            index=2,
            format_func=lambda x: ABILITY_COLUMNS[x],
            key="z_3d",
        )


    fig_3d = px.scatter_3d(
        df,

        x=x_3d,
        y=y_3d,
        z=z_3d,

        color="묶음",

        category_orders={
            "묶음": CIRCLE_LABELS[:cluster_count]
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
            x_3d: ABILITY_COLUMNS[x_3d],
            y_3d: ABILITY_COLUMNS[y_3d],
            z_3d: ABILITY_COLUMNS[z_3d],
            "묶음": "선수 유형",
        },

        title=(
            f"{ABILITY_COLUMNS[x_3d]} × "
            f"{ABILITY_COLUMNS[y_3d]} × "
            f"{ABILITY_COLUMNS[z_3d]}"
        ),
    )


    # 점을 작게
    fig_3d.update_traces(
        marker=dict(
            size=3
        )
    )

    # 3D 그래프 크게
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
        use_container_width=True
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
        index=CIRCLE_LABELS[:cluster_count],
        fill_value=0
    )
    .reindex(
        columns=[
            "공격수",
            "미드필더",
            "수비수",
            "기타",
        ],
        fill_value=0
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
        "클러스터링 방법: 선택한 능력치를 표준화한 후 K-Means 적용"
    )

    st.write(
        "난수 고정값: random_state = 42"
    )

    st.write(
        "묶음 이름: 슈팅 평균이 높은 묶음부터 "
        "㉮ → ㉯ → ㉰ → ㉱ → ㉲ → ㉳"
    )
