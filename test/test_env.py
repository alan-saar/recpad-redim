#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de teste para validar se o ambiente virtual e as dependências
foram instalados corretamente.
"""

import sys

def test_imports():
    print("=== Iniciando teste de sanidade das bibliotecas ===")
    print(f"Versão do Python: {sys.version}\n")
    
    libraries = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("matplotlib", "plt"),
        ("seaborn", "sns"),
        ("sklearn", "scikit-learn"),
        ("umap", "umap-learn")
    ]
    
    success = True
    for lib, name in libraries:
        try:
            __import__(lib)
            print(f"[ OK ] Biblioteca '{name}' importada com sucesso.")
        except ImportError as e:
            print(f"[ ERRO ] Falha ao importar '{name}': {e}")
            success = False
            
    if success:
        print("\n[ SUCESSO ] Todas as bibliotecas obrigatórias estão instaladas e funcionando perfeitamente!")
        sys.exit(0)
    else:
        print("\n[ FALHA ] Algumas bibliotecas estão faltando ou possuem problemas de instalação.")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
