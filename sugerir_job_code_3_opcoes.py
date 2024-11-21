import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import streamlit as st

# Baixar stopwords do NLTK (caso necessário)
nltk.download('stopwords')

# Ler o arquivo Excel hospedado no GitHub
url_excel = 'https://github.com/Diegosales01/Jobcode/blob/main/Base_Job_Code_2024.xlsx'
dados = pd.read_excel(url_excel)

# Obter stopwords em português
stop_words = stopwords.words('portuguese')

# Usar TfidfVectorizer com stop words, min_df ajustado e n-gramas
tfidf = TfidfVectorizer(stop_words=stop_words, min_df=1, ngram_range=(1, 2))

# Aplicar a vetorização ao conjunto de dados
matriz_tfidf = tfidf.fit_transform(dados['Descricao em 2024'])

# Configuração da interface com Streamlit
st.title("Sistema de Sugestão de Códigos de Cargos")
descricao_usuario = st.text_area("Digite a descrição do cargo:")

# Se o usuário inserir uma descrição, calculamos a similaridade
if descricao_usuario:
    # Vetorizar a descrição do usuário
    descricao_usuario_tfidf = tfidf.transform([descricao_usuario])
    
    # Calcular a similaridade de cosseno entre a descrição do usuário e as descrições do dataset
    similaridades = cosine_similarity(descricao_usuario_tfidf, matriz_tfidf)
    
    # Obter os índices das 3 descrições mais similares
    indices_similares = similaridades.argsort()[0, -3:][::-1]
    
    # Exibir as 3 opções de título e código
    st.write("Selecione o código mais adequado:")
    for i, indice in enumerate(indices_similares):
        descricao_similar = dados.iloc[indice]['Descricao em 2024']
        codigo_similar = dados.iloc[indice]['Job Code']
        titulo_similar = dados.iloc[indice]['Titulo em 2024']
        
        st.write(f"Opção {i+1}: {titulo_similar} - Código: {codigo_similar}")
        st.write(f"Descrição: {descricao_similar}")
        
        # O usuário seleciona o código mais adequado
        if st.button(f"Selecionar Opção {i+1}", key=i):
            st.write(f"Você selecionou o código: {codigo_similar}")
            
            # Função para registrar feedback do usuário
            def registrar_feedback(descricao_usuario, codigo_escolhido):
                feedback = {'Descricao em 2024': descricao_usuario, 'Job Code': codigo_escolhido}
                feedback_df = pd.DataFrame([feedback])
                
                # Salvar em um arquivo CSV para aprendizado futuro
                feedback_df.to_csv('feedback_usuario.csv', mode='a', index=False, header=False)
                st.write("Feedback registrado com sucesso!")

            # Registrar o feedback do usuário
            registrar_feedback(descricao_usuario, codigo_similar)
