import requests
from bs4 import BeautifulSoup
import time

# Lista de URLs a scrapear
urls = [
    "https://www.redmichoacan.com/2025/11/03/estudiantes-del-tecnologico-de-morelia-exigen-destitucion-de-la-directora-patricia-calderon-por-presunta-mala-gestion/",
    "https://www.respuesta.com.mx/noticias-principales/mantienen-alumnos-toma-del-tecnologico-de-morelia-no-se-levantaran-hasta-obtener-respuesta-conforme-a-sus-demandas/",
    "https://grupomarmor.com.mx/2025/11/04/estudiantes-del-tec-de-morelia-exigen-seguridad-transparencia-y-dignidad-en-sus-aulas/",
    "https://revistamorelia.com/paro-indefinido-en-el-tec-de-morelia/",
    "https://www.reddenoticiasmichoacan.com/noticia/estudiantes-del-itm-declaran-paro-indefinido",
    "https://estrellado.mx/Morelia/estudiantes-del-tec-de-morelia-suspenden-clases-en-protesta-por-asesinato-del-alcalde-de-uruapan"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

corpus = []

for i, url in enumerate(urls):
    try:
        print(f"Scrapeando {i+1}/{len(urls)}: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraer título (ajusta según estructura de cada sitio)
        title = soup.find('h1').get_text().strip() if soup.find('h1') else 'Título no encontrado'
        
        # Extraer contenido (busca selectores comunes de contenido)
        content = ""
        # Probamos diferentes selectores comunes para contenido
        content_selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            'article .content',
            'div[class*="content"]',
            'div[class*="entry"]'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text().strip() for elem in elements])
                break
        
        # Si no se encuentra con selectores, buscar todos los párrafos
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        corpus.append({
            'url': url,
            'title': title,
            'content': content
        })
        
        time.sleep(2)  # Espera entre requests
        
    except Exception as e:
        print(f"Error scrapeando {url}: {str(e)}")
        corpus.append({
            'url': url,
            'title': 'ERROR',
            'content': f'Error durante scraping: {str(e)}'
        })

# Guardar resultados
with open('corpus_noticias.txt', 'w', encoding='utf-8') as f:
    for noticia in corpus:
        f.write(f"URL: {noticia['url']}\n")
        f.write(f"Título: {noticia['title']}\n")
        f.write(f"Contenido: {noticia['content']}\n")
        f.write("\n" + "="*80 + "\n\n")

print("Corpus guardado en corpus_noticias.txt")