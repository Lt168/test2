import streamlit as st
import pickle
import pandas as pd

# 页面配置（恢复侧边栏导航，匹配图片布局）
st.set_page_config(
    page_title="医疗费用预测",
    page_icon="💰",
    layout="wide",  # 宽布局适配左右分栏
    initial_sidebar_state="expanded"  # 展开侧边栏
)


def introduce_page():
    """简介页面（匹配新图片样式）"""
    st.markdown("""
    <h1>欢迎使用</h1>
    <h2 style="color:#333;">医疗费用预测应用</h2>
    <p>这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。</p>

    <h3 style="margin-top: 40px; color:#333;">背景介绍</h3>
    <ul>
        <li>开发目标: 帮助保险公司合理定价保险产品，控制风险</li>
        <li>模型算法: 利用随机森林回归算法训练医疗费用预测模型</li>
    </ul>

    <h3 style="margin-top: 40px; color:#333;">使用指南</h3>
    <ul>
        <li>输入准确完整的被保险人信息，可以得到更准确的费用预测</li>
        <li>预测结果可以作为保险定价的重要参考，但需审慎决策</li>
        <li>有任何问题欢迎联系我们的技术支持</li>
    </ul>

    <p style="margin-top: 40px; color:#666;">技术支持: 📧 <a href="mailto:support@example.com">support@example.com</a></p>
    """, unsafe_allow_html=True)


def predict_page():
    """预测页面（匹配新图片样式）"""
    st.markdown("""
    <h2 style="color:#333;">使用说明</h2>
    <p>这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。</p>
    <ul>
        <li>输入信息: 在下面输入被保险人的个人信息、疾病信息等</li>
        <li>费用预测: 应用会预测被保险人的未来医疗费用支出</li>
    </ul>
    """, unsafe_allow_html=True)

    # 输入卡片（匹配图片的边框样式）
    with st.container(border=True):
        # 年龄输入（默认0，带+-按钮）
        age = st.number_input("年龄", min_value=0, value=0, step=1)
        
        # 性别单选框（默认选男性）
        sex = st.radio("性别", ["男性", "女性"], index=0)
        
        # BMI输入（默认0.00，带+-按钮）
        bmi = st.number_input("BMI", min_value=0.0, value=0.00, step=0.01)
        
        # 子女数量输入（默认0，带+-按钮）
        children = st.number_input("子女数量", min_value=0, value=0, step=1)
        
        # 是否吸烟单选框（默认选是）
        smoker = st.radio("是否吸烟", ["是", "否"], index=0)
        
        # 区域下拉框（默认选东南部）
        region = st.selectbox("区域", ["东南部", "西南部", "东北部", "西北部"], index=0)
        
        # 预测按钮（匹配图片样式）
        submit = st.button("预测费用", use_container_width=False)

    # 预测结果展示（保留原有逻辑）
    if submit:
        # 加载特征列
        try:
            with open("feature_columns.pkl", "rb") as f:
                feature_cols = pickle.load(f)
        except FileNotFoundError:
            st.error("请先运行save_model.py生成feature_columns.pkl！")
            return

        # 构造输入数据
        input_dict = {
            "年龄": age,
            "性别_男性": 1 if sex == "男性" else 0,
            "性别_女性": 1 if sex == "女性" else 0,
            "BMI": bmi,
            "子女数量": children,
            "是否吸烟_是": 1 if smoker == "是" else 0,
            "是否吸烟_否": 1 if smoker == "否" else 0,
            "区域_东南部": 1 if region == "东南部" else 0,
            "区域_西南部": 1 if region == "西南部" else 0,
            "区域_东北部": 1 if region == "东北部" else 0,
            "区域_西北部": 1 if region == "西北部" else 0
        }
        input_data = pd.DataFrame([input_dict], columns=feature_cols)

        # 预测逻辑
        try:
            with open("rfr_model.pkl", "rb") as f:
                model = pickle.load(f)
            pred = model.predict(input_data)[0]
            st.markdown(f"""
            <p style="margin-top:20px;">根据您输入的数据，预测该客户的医疗费用是: <span style="color:#333; font-weight:bold;">{round(pred, 2)}</span></p>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("请先运行save_model.py生成rfr_model.pkl！")
        except Exception as e:
            st.error(f"预测失败：{str(e)}")


# 侧边栏导航（匹配图片的单选按钮样式）
st.sidebar.markdown("<h5>导航</h5>", unsafe_allow_html=True)
nav = st.sidebar.radio(
    "",  # 隐藏默认标签
    ["简介", "预测医疗费用"],
    index=0  # 默认选中“简介”
)

# 页面跳转逻辑
if nav == "简介":
    introduce_page()
else:
    predict_page()
