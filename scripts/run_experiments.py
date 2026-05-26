#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de Experimentos - Trabalho 1: Redução de Dimensionalidade (PCA, t-SNE, UMAP)
Autor: Alan Saar & Leonardo Herkenhoff
Disciplina: Reconhecimento de Padrões (PPCOMP-IFES | 2026)

Este script realiza de forma automatizada:
1. Carregamento e pré-processamento (normalização e amostragem estratificada) do MNIST.
2. Execução dos métodos obrigatórios: PCA, t-SNE e UMAP.
3. Cálculo das métricas quantitativas: Trustworthiness, Continuity, Silhouette e Tempo.
4. Análise de estabilidade com 10 sementes aleatórias.
5. Varredura e avaliação de hiperparâmetros.
6. Avaliação Downstream usando classificador k-NN (Parte C).
7. Geração e exportação dos gráficos de alta qualidade para a pasta relatorio/img/.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Tentar importar o umap, se não estiver instalado exibirá aviso
try:
    import umap
except ImportError:
    print("[AVISO] A biblioteca 'umap-learn' não está instalada. Instale via pip.")

# Configurações de Estética das Figuras
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Pasta para salvar as imagens do relatório
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "relatorio", "img"))
os.makedirs(IMG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Carregamento e Pré-processamento
# ----------------------------------------------------------------------

def load_and_preprocess_mnist(n_samples=3000, random_state=42):
    """
    Carrega o MNIST, realiza amostragem estratificada por classes para manter a
    proporção dos dígitos (0-9) e normaliza os dados de pixel (0 a 1).
    """
    print(f"[*] Carregando o dataset MNIST (amostragem de {n_samples} instâncias)...")
    
    # Carregar o MNIST de forma rápida
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    # Amostragem Estratificada para manter a distribuição exata dos dígitos
    X_sample, _, y_sample, _ = train_test_split(
        X, y, 
        train_size=n_samples, 
        stratify=y, 
        random_state=random_state
    )
    
    # Normalização dos pixels para a escala [0, 1]
    X_scaled = X_sample / 255.0
    
    print(f"[ OK ] Dataset carregado com sucesso. Dimensões originais: {X_sample.shape}")
    print(f"       Distribuição das classes: {np.bincount(y_sample)}")
    
    return X_scaled, y_sample

# ----------------------------------------------------------------------
# 2. Métricas de Avaliação Customizadas
# ----------------------------------------------------------------------

def calculate_continuity(X_original, X_embedded, n_neighbors=15):
    """
    A métrica de Continuidade (Continuity) mede o quanto vizinhos próximos no
    espaço original são mantidos próximos no espaço projetado (falsas extrusões).
    Matematicamente, ela é o dual da Trustworthiness: equivale a calcular a
    Trustworthiness trocando a ordem das matrizes de alta e baixa dimensão.
    """
    from sklearn.manifold import trustworthiness
    # Continuidade é o inverso estrutural: avalia a preservação original no embutido.
    return trustworthiness(X_embedded, X_original, n_neighbors=n_neighbors)

def calculate_trustworthiness(X_original, X_embedded, n_neighbors=15):
    """
    Mede a fidelidade de vizinhança (falsas intrusões): pontos próximos na projeção
    eram de fato próximos no espaço original.
    """
    from sklearn.manifold import trustworthiness
    return trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)

def evaluate_embedding(X_original, X_embedded, y, duration, name, n_neighbors=15):
    """
    Calcula todas as métricas quantitativas para um embedding gerado.
    """
    print(f"[*] Calculando métricas quantitativas para {name}...")
    t0 = time.time()
    
    tw = calculate_trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)
    ct = calculate_continuity(X_original, X_embedded, n_neighbors=n_neighbors)
    sil = silhouette_score(X_embedded, y)
    
    metrics = {
        'Método': name,
        'Trustworthiness': tw,
        'Continuity': ct,
        'Silhouette Score': sil,
        'Tempo de Execução (s)': duration
    }
    
    print(f"    [ {name} ] Trustworthiness: {tw:.4f} | Continuity: {ct:.4f} | Silhouette: {sil:.4f} (Calculado em {time.time()-t0:.2f}s)")
    return metrics

# ----------------------------------------------------------------------
# 3. Execução dos Métodos e Geração de Gráficos
# ----------------------------------------------------------------------

def run_pca(X, n_components=2):
    """Executa a redução linear via PCA."""
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_emb = pca.fit_transform(X)
    duration = time.time() - t0
    return X_emb, duration

def run_tsne(X, perplexity=30, learning_rate=200, n_iter=1000, random_state=42):
    """Executa a redução não linear via t-SNE."""
    # O scikit-learn recomenda PCA preliminar (50 componentes) antes do t-SNE
    # para acelerar a computação e suprimir ruído.
    t0 = time.time()
    if X.shape[1] > 50:
        pca_pre = PCA(n_components=50, random_state=random_state)
        X_input = pca_pre.fit_transform(X)
    else:
        X_input = X
        
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        learning_rate=learning_rate, 
        max_iter=n_iter, 
        random_state=random_state,
        n_jobs=-1
    )
    X_emb = tsne.fit_transform(X_input)
    duration = time.time() - t0
    return X_emb, duration

def run_umap(X, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """Executa a redução não linear via UMAP."""
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        metric=metric, 
        random_state=random_state,
        n_jobs=-1
    )
    X_emb = reducer.fit_transform(X)
    duration = time.time() - t0
    return X_emb, duration

# ----------------------------------------------------------------------
# 4. Plots de Embeddings
# ----------------------------------------------------------------------

def plot_embedding(X_emb, y, title, filename):
    """Gera e salva o gráfico de dispersão colorido do embedding."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_emb[:, 0], X_emb[:, 1], 
        c=y, cmap='tab10', 
        s=5, alpha=0.7, 
        edgecolors='none'
    )
    plt.colorbar(scatter, ticks=range(10), label="Dígito")
    plt.title(title)
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.tight_layout()
    
    path = os.path.join(IMG_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"[ OK ] Gráfico salvo em: {path}")

# ----------------------------------------------------------------------
# 5. Experimento de Estabilidade (Parte B - 10 Sementes)
# ----------------------------------------------------------------------

def run_stability_experiment(X, y, n_seeds=10):
    """
    Roda os algoritmos estocásticos (t-SNE e UMAP) sob 'n_seeds' sementes distintas
    para avaliar a reprodutibilidade e a variabilidade das projeções e métricas.
    """
    print(f"\n[*] Iniciando Experimento de Estabilidade com {n_seeds} sementes distintas...")
    
    seeds = [10 * i + 42 for i in range(n_seeds)]
    results = []
    
    for i, seed in enumerate(seeds):
        print(f"--- Semente {i+1}/{n_seeds} (seed={seed}) ---")
        
        # t-SNE
        tsne_emb, tsne_dur = run_tsne(X, random_state=seed)
        tsne_metrics = evaluate_embedding(X, tsne_emb, y, tsne_dur, "t-SNE", n_neighbors=15)
        tsne_metrics['Semente'] = seed
        results.append(tsne_metrics)
        
        # Salva o gráfico das 3 primeiras sementes para mostrar variação visual no artigo
        if i < 3:
            plot_embedding(tsne_emb, y, f"t-SNE (Semente {seed})", f"stability_tsne_seed_{seed}.png")
            
        # UMAP
        umap_emb, umap_dur = run_umap(X, random_state=seed)
        umap_metrics = evaluate_embedding(X, umap_emb, y, umap_dur, "UMAP", n_neighbors=15)
        umap_metrics['Semente'] = seed
        results.append(umap_metrics)
        
        if i < 3:
            plot_embedding(umap_emb, y, f"UMAP (Semente {seed})", f"stability_umap_seed_{seed}.png")
            
    df_stability = pd.DataFrame(results)
    
    # Gerar Estatística Descritiva (Média e Desvio Padrão)
    summary = df_stability.groupby('Método').agg(['mean', 'std']).round(4)
    print("\n=== Resumo de Estabilidade (Média ± Desvio Padrão) ===")
    print(summary)
    
    # Salvar tabela formatada para o Typst
    summary_path = os.path.join(IMG_DIR, "stability_summary.csv")
    summary.to_csv(summary_path)
    print(f"[ OK ] Tabela de estabilidade salva em: {summary_path}")
    
    # Gerar Boxplot de estabilidade das métricas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ['Trustworthiness', 'Continuity', 'Silhouette Score']
    for idx, metric in enumerate(metrics_to_plot):
        sns.boxplot(data=df_stability, x='Método', y=metric, ax=axes[idx], palette="Set2")
        axes[idx].set_title(f"Estabilidade de {metric}")
        axes[idx].set_ylabel(metric)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "stability_boxplot.png"))
    plt.close()
    print(f"[ OK ] Boxplot de estabilidade salvo em: {os.path.join(IMG_DIR, 'stability_boxplot.png')}")
    
    return df_stability

# ----------------------------------------------------------------------
# 6. Varredura de Hiperparâmetros
# ----------------------------------------------------------------------

def run_hyperparameter_sweep(X, y):
    """
    Avalia o impacto dos hiperparâmetros de t-SNE e UMAP na projeção final
    tanto qualitativa quanto quantitativamente.
    """
    print("\n[*] Iniciando Varredura de Hiperparâmetros...")
    
    # 6.1 t-SNE: Perplexity
    perplexities = [5, 30, 100]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, perp in enumerate(perplexities):
        print(f"[*] Executando t-SNE com perplexity = {perp}...")
        emb, dur = run_tsne(X, perplexity=perp, random_state=42)
        
        scatter = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=2, alpha=0.6)
        axes[idx].set_title(f"t-SNE (Perplexity = {perp})")
        axes[idx].axis('off')
        
        # Salva o plot individual também
        plot_embedding(emb, y, f"t-SNE (Perplexity = {perp})", f"tsne_perp_{perp}.png")
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "tsne_hyperparameter_sweep.png"))
    plt.close()
    print("[ OK ] Painel de perplexidade t-SNE salvo.")
    
    # 6.2 UMAP: n_neighbors
    neighbors = [5, 15, 80]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, nn in enumerate(neighbors):
        print(f"[*] Executando UMAP com n_neighbors = {nn}...")
        emb, dur = run_umap(X, n_neighbors=nn, random_state=42)
        
        scatter = axes[idx].scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=2, alpha=0.6)
        axes[idx].set_title(f"UMAP (n_neighbors = {nn})")
        axes[idx].axis('off')
        
        # Salva o plot individual também
        plot_embedding(emb, y, f"UMAP (n_neighbors = {nn})", f"umap_neighbors_{nn}.png")
        
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "umap_hyperparameter_sweep.png"))
    plt.close()
    print("[ OK ] Painel de n_neighbors UMAP salvo.")

# ----------------------------------------------------------------------
# 7. Avaliação Downstream (Parte C)
# ----------------------------------------------------------------------

def run_downstream_evaluation(X, y):
    """
    Treina um classificador k-NN (k=5) nos dados originais (alta dimensão)
    e compara o seu desempenho com os embeddings de 2 dimensões do PCA, t-SNE e UMAP.
    """
    print("\n[*] Iniciando Avaliação Downstream (Classificação k-NN)...")
    
    # Split de Treino e Teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 1. k-NN nos dados originais de alta dimensão (784D)
    knn_orig = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_orig.fit(X_train, y_train)
    y_pred_orig = knn_orig.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    dur_orig = time.time() - t0
    print(f"    [ Alta Dimensão (784D) ] Acurácia k-NN: {acc_orig:.4f} (Tempo: {dur_orig:.2f}s)")
    
    results = [{
        'Método': 'Espaço Original (784D)',
        'Dimensões': 784,
        'Acurácia k-NN (k=5)': acc_orig,
        'Tempo de Classificação (s)': dur_orig
    }]
    
    # 2. k-NN no PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_pca.fit(X_tr_pca, y_train)
    y_pred_pca = knn_pca.predict(X_te_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    dur_pca = time.time() - t0
    print(f"    [ PCA (2D) ] Acurácia k-NN: {acc_pca:.4f} (Tempo: {dur_pca:.2f}s)")
    
    results.append({
        'Método': 'PCA (2D)',
        'Dimensões': 2,
        'Acurácia k-NN (k=5)': acc_pca,
        'Tempo de Classificação (s)': dur_pca
    })
    
    # 3. k-NN no t-SNE 2D
    # Nota: t-SNE não permite aplicar .transform() diretamente em dados novos, 
    # pois ele é um método transdutivo. Portanto, geramos o embedding global 
    # de todo o conjunto de dados e dividimos o embedding gerado em treino/teste.
    tsne_emb, _ = run_tsne(X, random_state=42)
    X_tr_tsne, X_te_tsne, _, _ = train_test_split(
        tsne_emb, y, test_size=0.2, stratify=y, random_state=42
    )
    
    knn_tsne = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_tsne.fit(X_tr_tsne, y_train)
    y_pred_tsne = knn_tsne.predict(X_te_tsne)
    acc_tsne = accuracy_score(y_test, y_pred_tsne)
    dur_tsne = time.time() - t0
    print(f"    [ t-SNE (2D) ] Acurácia k-NN: {acc_tsne:.4f} (Tempo: {dur_tsne:.2f}s)")
    
    results.append({
        'Método': 't-SNE (2D)',
        'Dimensões': 2,
        'Acurácia k-NN (k=5)': acc_tsne,
        'Tempo de Classificação (s)': dur_tsne
    })
    
    # 4. k-NN no UMAP 2D
    # UMAP suporta transform() em novos dados.
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
    X_tr_umap = reducer.fit_transform(X_train)
    X_te_umap = reducer.transform(X_test)
    
    knn_umap = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_umap.fit(X_tr_umap, y_train)
    y_pred_umap = knn_umap.predict(X_te_umap)
    acc_umap = accuracy_score(y_test, y_pred_umap)
    dur_umap = time.time() - t0
    print(f"    [ UMAP (2D) ] Acurácia k-NN: {acc_umap:.4f} (Tempo: {dur_umap:.2f}s)")
    
    results.append({
        'Método': 'UMAP (2D)',
        'Dimensões': 2,
        'Acurácia k-NN (k=5)': acc_umap,
        'Tempo de Classificação (s)': dur_umap
    })
    
    # Salvar resultados em DataFrame e exportar
    df_knn = pd.DataFrame(results)
    knn_path = os.path.join(IMG_DIR, "knn_downstream_results.csv")
    df_knn.to_csv(knn_path, index=False)
    print(f"[ OK ] Tabela de avaliação downstream salva em: {knn_path}")
    
    # Gerar Gráfico de Acurácia k-NN
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_knn, x='Método', y='Acurácia k-NN (k=5)', palette='viridis')
    plt.title("Acurácia de Classificação k-NN (k=5) nos Espaços Reduzidos vs. Original")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1.05)
    
    # Adicionar os valores acima das barras
    for index, row in df_knn.iterrows():
        plt.text(index, row['Acurácia k-NN (k=5)'] + 0.01, f"{row['Acurácia k-NN (k=5)']:.4f}", 
                 color='black', ha="center", fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "knn_downstream_bar.png"))
    plt.close()
    print(f"[ OK ] Gráfico de barras downstream salvo em: {os.path.join(IMG_DIR, 'knn_downstream_bar.png')}")

# ----------------------------------------------------------------------
# 8. Execução Principal
# ----------------------------------------------------------------------

def main():
    print("======================================================================")
    print("INICIANDO PIPELINE DE EXPERIMENTOS DE REDUÇÃO DE DIMENSIONALIDADE")
    print("======================================================================")
    
    # 1. Carregamento dos dados
    X, y = load_and_preprocess_mnist(n_samples=3000, random_state=42)
    
    # 2. Execução Básica (Parte A - Fidelidade Estrutural)
    print("\n[*] Rodando métodos básicos...")
    
    # PCA
    pca_emb, pca_dur = run_pca(X)
    pca_metrics = evaluate_embedding(X, pca_emb, y, pca_dur, "PCA", n_neighbors=15)
    plot_embedding(pca_emb, y, "PCA - Projeção 2D (MNIST)", "projection_pca.png")
    
    # t-SNE
    tsne_emb, tsne_dur = run_tsne(X, random_state=42)
    tsne_metrics = evaluate_embedding(X, tsne_emb, y, tsne_dur, "t-SNE", n_neighbors=15)
    plot_embedding(tsne_emb, y, "t-SNE - Projeção 2D (MNIST)", "projection_tsne.png")
    
    # UMAP
    umap_emb, umap_dur = run_umap(X, random_state=42)
    umap_metrics = evaluate_embedding(X, umap_emb, y, umap_dur, "UMAP", n_neighbors=15)
    plot_embedding(umap_emb, y, "UMAP - Projeção 2D (MNIST)", "projection_umap.png")
    
    # Consolidar Tabela da Parte A
    df_metrics = pd.DataFrame([pca_metrics, tsne_metrics, umap_metrics])
    metrics_path = os.path.join(IMG_DIR, "structural_metrics.csv")
    df_metrics.round(4).to_csv(metrics_path, index=False)
    print(f"\n[ OK ] Tabela de fidelidade estrutural salva em: {metrics_path}")
    print(df_metrics.round(4))
    
    # 3. Estabilidade (Parte B - 10 sementes)
    run_stability_experiment(X, y, n_seeds=10)
    
    # 4. Varredura de Hiperparâmetros (Parte B - Parâmetros)
    run_hyperparameter_sweep(X, y)
    
    # 5. Avaliação Downstream (Parte C - k-NN)
    run_downstream_evaluation(X, y)
    
    print("\n======================================================================")
    print("[ SUCESSO ] Todos os experimentos foram executados e os resultados foram salvos!")
    print("======================================================================")

if __name__ == "__main__":
    main()
