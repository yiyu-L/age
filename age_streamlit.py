import pandas as pd
from streamlit import caching
import numpy as np 
import math
import plotly.express as px
import streamlit as st 
import base64
import io

@st.cache(persist=False, allow_output_mutation=True, suppress_st_warning=False, show_spinner= False)
def load_data():
    df_input = pd.DataFrame()
    df_input = pd.read_excel(input, engine = 'openpyxl', sheet_name = 0)
    return df_input

def age_trans(x):
    if isinstance(x, str):
        if x[-1] in ['Y', 'y', '岁'] and len(x) > 1:
            return x[:-1].strip()
        if x[-1] in ['M', 'm', '月'] and len(x) > 1:
            return math.ceil( int( x[:-1].strip() ) / 12)
        if x[-1] in ['D', 'd', '天'] and len(x) > 1:
            return math.ceil( int( x[:-1].strip() ) / 365)
        if x[-1] in ['W', 'w', '周'] and len(x) > 1:
            return math.ceil( int( x[:-1].strip() ) * 7 / 365)
        if x.isdigit():
            return x
        else:
            return -1
    else:
        return -1

def ge_120(y):
    if y > 120:
        return -1
    else:
        return y

def tb_in(x):
    if '活动性肺结核' in x:
        return 1
    else:
        return 0

def prep_data(df):
    df['年龄Y'] = df['年龄'].apply(age_trans).astype(float).apply(np.ceil).apply(ge_120)
    df['tb_标注'] = df['医生审核结果'].apply(tb_in).astype(int)
    df['tb分值'] = df['AI分值']
    df = df[['年龄', 'AI分值', '医生审核结果', '年龄Y', 'tb_标注', 'tb分值']]
    return df

def AI_threshold_set(df):
    df['tb_AI'] = (df['tb分值'] > AI_threshold*0.01).astype(int)
    return df

def age_bar(df):
    df_age = df['年龄Y'].value_counts()
    fig = px.bar(df_age, labels = {'index':'年龄','value': '数量'}, title = '年龄分布')
    return fig

def thre_compute(x):
    total = len( x )
    tp = ( ( x['tb_标注'] ==1) & ( x['tb_AI'] == 1 ) ).astype(int).sum()  # 真阳
    tn = ( ( x['tb_标注'] ==0) & ( x['tb_AI'] == 0 ) ).astype(int).sum()  # 真阴
    fn = ( ( x['tb_标注'] ==1) & ( x['tb_AI'] == 0 ) ).astype(int).sum()  # 假阴
    fp = ( ( x['tb_标注'] ==0) & ( x['tb_AI'] == 1 ) ).astype(int).sum()  # 假阳
    p_num = tp + fn # 阳性数量
    fn_rate = fn / total # 假阴率
    fp_rate = fp / (fp + tp) # 假阳率
    p_fn_rate = fn / p_num  # 阳性患者假阴率
    return total, p_num, tp, tn, fn, fp, fn_rate, fp_rate, p_fn_rate

def result_output(df):
    mat_up = []
    for threshold_age in range(-1, 70):
        mat_up.append([threshold_age, *thre_compute(df[df['年龄Y'] >= threshold_age])])
    mat_low = []
    mat_low.append([-1, *thre_compute(df[df['年龄Y'] == -1])])
    for threshold_age in range(70):
        mat_low.append([threshold_age, *thre_compute( df[(df['年龄Y'] <= threshold_age)&(df['年龄Y'] >= 0)])])
    mat = np.hstack( (  np.array(mat_up), np.array(mat_low)  ) )
    matr = pd.DataFrame( mat, columns = [['年龄阈值及以上的分析(包含年龄阈值)']*10 + ['年龄阈值及以下的分析(包含年龄阈值)'] * 10, ['年龄阈值', '样本数量', '阳性数量', '真阳', '真阴', '假阴', '假阳','假阴率', '假阳率', '阳性患者假阴率']*2 ] )
    return matr


if __name__ == "__main__":
    st.set_page_config( page_title="年龄阈值分析", page_icon = '👨‍🏭', initial_sidebar_state="collapsed", layout='wide' )
    
    st.title('年龄阈值分析')

    caching.clear_cache()
    df = pd.DataFrame()

    st.subheader('1.数据加载')
    st.write("请上传一个xlsx文件")
    with st.beta_expander('数据格式'):
        mat_format = [['年龄','短文本','接受数字、以“Y/M/W/D/y/m/w/d/岁/月/周/天"结尾的文本，不在上述类型中的文本均被认为是年龄未知'], ['AI分值','数字','无'],['医生审核结果','短文本','含有“活动性肺结核”字样的文本，才认为是确诊。'],['无','无','建议增加一列附上数据的路径，如：studyUid、DCM文本路径'] ]
        df_format = pd.DataFrame(mat_format, columns = ['字段名称','数据类型','说明'] )
        st.table(df_format)
    input = st.file_uploader('请选择一个文件')
    if input is None:
            sample = st.checkbox("从GitHub上下载demo.xlsx")
    try:
        if sample:
            st.markdown("""[下载链接](https://github.com/yiyu-L/age)""")    
    except:
       if input:
           with st.spinner('加载数据中...'):
                df = load_data()
    with st.beta_expander('显示原始数据'):
      st.write(df)
    

    st.subheader('2.年龄分布图')
    if not df.empty:
        df1 = prep_data(df)
        df1 = pd.DataFrame(df1)
        radar_chart_fig = age_bar(df1)
        st.plotly_chart(radar_chart_fig)

    st.subheader('3.不同年龄阈值的结果')
    AI_threshold = st.slider("AI阈值设置(%)", 0, 100)
    if not df.empty:
        df2 = AI_threshold_set(df1)
        result = result_output(df2)
        st.write(result)
        df_export = pd.DataFrame(result)
        output = io.BytesIO()
        df_export.to_excel(output, engine = 'openpyxl') 
        b64 = base64.b64encode(output.getvalue())
        href = f'<a download = "年龄阈值分析结果.xlsx" href="data:file/xlsx;base64,{b64}">Download XLSX File</a> (**年龄阈值分析结果.xlsx**)'
        st.markdown(href, unsafe_allow_html=True)
