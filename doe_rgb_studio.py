#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoE RGB Studio - leitura colorimetrica de tubos por foto.

Abas
  1. Planejamento     - abre o projeto do Minitab (.mpx) ou uma planilha (CSV/XLSX),
                        mostra os ensaios e aceita a concentracao de cada um.
  2. Fotos e ROIs     - abre as fotos, propoe uma ROI retangular por tubo e deixa
                        ajustar vertices, mover, criar, apagar (com desfazer/refazer).
  3. Sinal e calibracao - A = -log10(sinal do tubo / branco do tubo), curva de
                        calibracao, grafico e exportacao em XLSX.

Dependencias:
  pip install opencv-python numpy pandas openpyxl pillow matplotlib
"""
import colorsys
import json
import math
import os

import cv2
import numpy as np
import pandas as pd

LARGURA_REF = 900
LADO_PADRAO = 28
REF_DIST = 35
REF_LADO = 40
RESPOSTAS = ["A_R", "A_G", "A_B", "norma_A", "R", "G", "B", "H", "S", "V", "I"]
AZUL, LARANJA, TINTA, SEC, FUNDO = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def ler_imagem(caminho):
    dados = np.fromfile(caminho, dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Nao foi possivel abrir a imagem: {caminho}")
    return img


def num(v):
    try:
        return float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return float("nan")


def id_ensaio(v):
    s = str(v).strip()
    try:
        f = float(s.replace(",", "."))
        return str(int(f)) if f == int(f) else s
    except ValueError:
        return s.upper() if s[:1].lower() == "b" and s[1:].isdigit() else s


def ordenar_ids(ids):
    """Ordena IDs numericos como numeros ('2' antes de '10') e o resto em ordem alfabetica."""
    def chave(i):
        v = num(i)
        return (0, v, "") if not math.isnan(v) else (1, 0.0, str(i))
    return sorted(ids, key=chave)


def partes_do_arquivo(caminho):
    base = os.path.splitext(os.path.basename(caminho))[0]
    for sep in ("_", ",", ";", " "):
        base = base.replace(sep, "-")
    return [id_ensaio(p) for p in base.split("-") if p.strip()]


# ============================================================================
# GEOMETRIA DAS ROIs (poligonos: retangulo por padrao, vertices editaveis)
# ============================================================================
def retangulo(cx, cy, lado_x, lado_y=None):
    ly = lado_y if lado_y is not None else lado_x
    hx, hy = lado_x / 2.0, ly / 2.0
    return [[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]]


def centro_roi(p):
    v = np.asarray(p["pts"], float)
    return float(v[:, 0].mean()), float(v[:, 1].mean())


def dentro(p, x, y):
    v = np.asarray(p["pts"], np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(v, (float(x), float(y)), False) >= 0


def dist_segmento(x, y, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(x - px, y - py), [px, py]


def aresta_mais_perto(p, x, y):
    """(indice do vertice depois do qual inserir, distancia, ponto na aresta)."""
    pts = p["pts"]
    melhor = (0, 1e9, [x, y])
    for i in range(len(pts)):
        d, q = dist_segmento(x, y, pts[i], pts[(i + 1) % len(pts)])
        if d < melhor[1]:
            melhor = (i, d, q)
    return melhor


def area_roi(p):
    v = np.asarray(p["pts"], np.float32)
    return float(abs(cv2.contourArea(v)))


# ============================================================================
# MEDICAO
# ============================================================================
def _branco_abaixo(img, x, y_fundo, esc):
    h, w = img.shape[:2]
    lado = max(8, int(REF_LADO * esc))
    y0 = int(min(h - lado - 1, y_fundo + REF_DIST * esc))
    x0 = int(np.clip(x - lado // 2, 0, w - lado - 1))
    patch = img[y0:y0 + lado, x0:x0 + lado].reshape(-1, 3)
    bgr = np.median(patch, axis=0)
    return bgr[::-1].astype(float), (x0, y0, lado, lado)


def rgb_do_branco(img, x, y, lado=None):
    k = max(4, int((lado or 20) / 2 * img.shape[1] / LARGURA_REF))
    x, y = int(x), int(y)
    patch = img[max(0, y - k):y + k, max(0, x - k):x + k].reshape(-1, 3)
    if not len(patch):
        return None
    return [float(v) for v in np.median(patch, axis=0)[::-1]]


def medir_roi(img, pts, branco_rgb=None):
    """Mede a cor media dentro do poligono. Retorna (resultados, mascara booleana, ret_branco)."""
    h, w = img.shape[:2]
    esc = w / LARGURA_REF
    masc = np.zeros((h, w), np.uint8)
    poly = np.array([[int(round(x)), int(round(y))] for x, y in pts], np.int32)
    cv2.fillPoly(masc, [poly], 1)
    m = masc.astype(bool)
    if not m.any():
        cx, cy = poly.mean(axis=0)
        cv2.circle(masc, (int(cx), int(cy)), max(2, int(3 * esc)), 1, -1)
        m = masc.astype(bool)
    px = img[m][:, ::-1].astype(float)
    media = px.mean(axis=0)
    dp = px.std(axis=0, ddof=1) if len(px) > 1 else np.zeros(3)
    ys, xs = np.where(m)
    if branco_rgb is None:
        branco, ret_branco = _branco_abaixo(img, int(xs.mean()), int(ys.max()), esc)
    else:
        branco, ret_branco = np.asarray(branco_rgb, float), None
    norm = np.clip(media / np.maximum(branco, 1) * 255.0, 0, 255)
    absorb = -np.log10(np.clip(media / np.maximum(branco, 1), 1e-3, None))
    hh, ss, vv = colorsys.rgb_to_hsv(*(media / 255.0))
    _, ll, _ = colorsys.rgb_to_hls(*(media / 255.0))
    res = {
        "n_pixels": int(m.sum()), "area_px": int(m.sum()),
        "R": media[0], "G": media[1], "B": media[2],
        "DP_R": dp[0], "DP_G": dp[1], "DP_B": dp[2],
        "H": hh * 360.0, "S": ss * 100.0, "V": vv * 100.0,
        "L": ll * 100.0, "I": media.mean(),
        "R_branco": branco[0], "G_branco": branco[1], "B_branco": branco[2],
        "R_norm": norm[0], "G_norm": norm[1], "B_norm": norm[2],
        "A_R": absorb[0], "A_G": absorb[1], "A_B": absorb[2],
        "norma_A": float(np.linalg.norm(absorb)),
    }
    return res, m, ret_branco


def medir_foto(img, rois, nomes, branco_rgb=None, branco_xy=None):
    """Mede todas as ROIs de uma foto e devolve (linhas, imagem anotada)."""
    anot = img.copy()
    esc = img.shape[1] / LARGURA_REF
    esp = max(1, int(round(2 * esc)))
    linhas = []
    for p, nome in zip(rois, nomes):
        res, masc, ret = medir_roi(img, p["pts"], branco_rgb)
        linhas.append({"ID": nome, **res})
        poly = np.array([[int(round(x)), int(round(y))] for x, y in p["pts"]], np.int32)
        cv2.polylines(anot, [poly], True, (0, 220, 255), esp)
        if ret:
            rx, ry, rw, rh = ret
            cv2.rectangle(anot, (rx, ry), (rx + rw, ry + rh), (0, 160, 0), esp)
        rot = str(nome)
        (tw, th), _ = cv2.getTextSize(rot, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * esc, esp)
        px0, py0 = int(poly[:, 0].mean() - tw / 2), int(poly[:, 1].min() - 8 * esc)
        cv2.rectangle(anot, (px0 - 3, py0 - th - 3), (px0 + tw + 3, py0 + 4), (255, 255, 255), -1)
        cv2.putText(anot, rot, (px0, py0), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * esc,
                    (0, 0, 200), esp, cv2.LINE_AA)
    if branco_xy is not None:
        bx, by = map(int, branco_xy)
        k = int(10 * esc)
        cv2.rectangle(anot, (bx - k, by - k), (bx + k, by + k), (0, 160, 0), esp)
        cv2.putText(anot, "branco", (bx + k + 4, by + k), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * esc,
                    (0, 130, 0), esp, cv2.LINE_AA)
    return linhas, anot


# ============================================================================
# DETECCAO AUTOMATICA DAS ROIs (devolve caixas retangulares)
# ============================================================================
def detectar_rois(img, n=None, lado_ref=LADO_PADRAO):
    h0, w0 = img.shape[:2]
    esc = w0 / LARGURA_REF
    p = cv2.resize(img, (LARGURA_REF, max(1, int(round(h0 / esc)))), interpolation=cv2.INTER_AREA)
    p = cv2.GaussianBlur(p, (0, 0), 1.2)
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    croma = np.hypot(a, b)
    fundo = (L > np.percentile(L, 55)) & (croma < np.percentile(croma, 50))
    dc = np.hypot(a - np.median(a[fundo]), b - np.median(b[fundo]))
    H, W = dc.shape
    forte = (dc > 12).astype(np.uint8)
    nl, rot, st, _ = cv2.connectedComponentsWithStats(forte, connectivity=8)
    grande = np.isin(rot, [i for i in range(1, nl)
                           if st[i, cv2.CC_STAT_AREA] > 0.02 * H * W]).astype(np.uint8)
    peq = cv2.resize(dc, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    aberto = cv2.morphologyEx(peq, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)))
    aberto = cv2.resize(cv2.GaussianBlur(aberto, (0, 0), 3), (W, H))
    th = np.clip(dc - aberto, 0, None)
    cand = (th > 6).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 5)
    nl, comp, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    nucleo = np.zeros_like(cand)
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] >= 250:
            m = comp == i
            nucleo[m & (dist > max(6.0, 0.55 * dist[m].max()))] = 1
    nn, nuc = cv2.connectedComponents(nucleo, connectivity=8)
    centros = []
    if nn > 1:
        for i in range(1, nn):
            iy, ix = np.unravel_index(np.argmax(np.where(nuc == i, dist, 0)), dist.shape)
            centros.append((ix, iy))
    ys, xs = np.nonzero(cand)
    mais_perto = None
    if centros and len(xs) * len(centros) < 4e7:
        cx = np.array([c[0] for c in centros])
        cy = np.array([c[1] for c in centros])
        mais_perto = np.argmin((xs[:, None] - cx[None]) ** 2 + (ys[:, None] - cy[None]) ** 2, axis=1)

    def coletar(area_max, b_max, usar_grande):
        """Candidatos aceitos com um dado conjunto de limites de tamanho."""
        saida = []
        if mais_perto is None:
            return saida
        for j, (ix, iy) in enumerate(centros):
            sel = (mais_perto == j) & (comp[ys, xs] == comp[iy, ix])
            px, py = xs[sel], ys[sel]
            area = len(px)
            if area < 250 or area > area_max:
                continue
            bw, bh = px.max() - px.min() + 1, py.max() - py.min() + 1
            if bw > b_max or bh > b_max or bw < 18 or bh < 12 or area / (bw * bh) < 0.40:
                continue
            if px.min() <= 1 or py.min() <= 1 or px.max() >= W - 2 or py.max() >= H - 2:
                continue
            if usar_grande and grande[py, px].mean() > 0.3:
                continue
            saida.append({"x": float(ix * esc), "y": float(iy * esc),
                          "x0": float(px.min() * esc), "x1": float(px.max() * esc),
                          "y0": float(py.min() * esc), "y1": float(py.max() * esc),
                          "peso": float(area * th[py, px].mean()), "area": int(area)})
        return saida

    # 1o passe: tubos lado a lado numa bancada. 2o passe (so se nada for achado):
    # foto de perto, com um ou dois tubos ocupando boa parte da imagem.
    ach = coletar(12000, 190, True) or coletar(0.25 * H * W, 600, False)
    if ach:
        banda, limite = 170 * esc, (n or 8)
        melhor, nota_melhor = [], -1.0
        for anc in ach:
            fila = []
            for d in sorted([d for d in ach if abs(d["y"] - anc["y"]) < banda],
                            key=lambda d: d["peso"], reverse=True):
                if all(abs(d["x"] - u["x"]) > 45 * esc for u in fila):
                    fila.append(d)
                if len(fila) == limite:
                    break
            ys_ = np.array([d["y"] for d in fila]) / esc
            ar = np.array([d["area"] for d in fila], float)
            alinh = math.exp(-float(ys_.std()) / 40.0)
            tam = float(ar.min() / ar.max()) ** 0.5
            pos = min(1.0, 0.3 + 0.7 * min(ys_.mean(), H - ys_.mean()) / (0.15 * H))
            nota = sum(d["peso"] for d in fila) * alinh * tam * pos
            if nota > nota_melhor:
                melhor, nota_melhor = fila, nota
        ach = melhor
    ach = sorted(ach, key=lambda d: d["x"])
    saida = []
    for d in ach:
        mx, my = 0.12 * (d["x1"] - d["x0"]), 0.12 * (d["y1"] - d["y0"])
        x0, x1 = d["x0"] + mx, d["x1"] - mx
        y0, y1 = d["y0"] + my, d["y1"] - my
        if x1 - x0 < 4 * esc or y1 - y0 < 4 * esc:
            x0, y0, x1, y1 = (d["x"] - lado_ref * esc / 2, d["y"] - lado_ref * esc / 2,
                              d["x"] + lado_ref * esc / 2, d["y"] + lado_ref * esc / 2)
        saida.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return saida


def branco_automatico(img, tubos=()):
    h0, w0 = img.shape[:2]
    esc = w0 / LARGURA_REF
    p = cv2.resize(img, (LARGURA_REF, max(1, int(round(h0 / esc)))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, croma = lab[..., 0], np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    k = 41
    mL, mC = cv2.blur(L, (k, k)), cv2.blur(croma, (k, k))
    dp = np.sqrt(np.clip(cv2.blur(L * L, (k, k)) - mL ** 2, 0, None))
    estourado = cv2.blur((p.max(axis=2) >= 250).astype(np.float32), (k, k))
    nota = mL - 3.0 * dp - 2.0 * mC - 200.0 * estourado
    H, W = nota.shape
    valido = np.zeros((H, W), bool)
    valido[30:H - 30, 30:W - 30] = True
    if tubos:
        tx = np.array([t[0] for t in tubos]) / esc
        ty = np.array([t[1] for t in tubos]) / esc
        faixa = np.zeros((H, W), bool)
        faixa[int(min(H - 1, ty.max() + 70)):int(min(H, ty.max() + 330)),
              int(max(0, tx.min() - 150)):int(min(W, tx.max() + 150))] = True
        if (valido & faixa).sum() > 2000:
            valido &= faixa
        for x, y in zip(tx, ty):
            cv2.circle(nota, (int(x), int(y)), 110, -1e9, -1)
    nota = np.where(valido, nota, -1e9)
    _, vmax, _, pos = cv2.minMaxLoc(nota)
    return None if vmax < -1e8 else (float(pos[0] * esc), float(pos[1] * esc))


# ============================================================================
# PLANEJAMENTO: MINITAB (.mpx) OU PLANILHA (CSV / XLSX)
# ============================================================================
LETRAS = "ABCDEFGHJKLM"
AUSENTE_MINITAB = 1.0e29
PAPEIS = {"padrao": {"ordempad", "stdorder", "ordenest", "ordemp", "ordempadrao"},
          "execucao": {"ordemens", "runorder", "ordencorrida", "ordemensaio"},
          "centro": {"ptcentral", "centerpt", "ptocentral"},
          "blocos": {"blocos", "blocks", "bloques"},
          "tipo": {"tipopt", "pttype", "tipoponto"}}
LINHA = "(linha da planilha)"
SEM_CONC = "(digitar manualmente)"


def _chave(nome):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode().lower()
    return "".join(c for c in s if c.isalnum())


def ler_mpx(caminho):
    import re
    import zipfile
    try:
        z = zipfile.ZipFile(caminho)
    except (zipfile.BadZipFile, OSError) as e:
        raise ValueError("O arquivo nao e um projeto .mpx do Minitab valido. Projetos antigos "
                         "(.mpj/.mtw) precisam ser salvos como .mpx, ou exporte em CSV/XLSX.") from e
    nomes = z.namelist()

    def ler(sufixo):
        alvo = [n for n in nomes if n.lstrip("/") == sufixo.lstrip("/")]
        if not alvo:
            raise ValueError(f"Estrutura inesperada no .mpx (nao encontrei {sufixo}).")
        return json.loads(z.read(alvo[0]).decode("utf-8-sig"))

    meta_nome = [n for n in nomes if n.lstrip("/").startswith("project_metadata") and n.endswith(".json")]
    if not meta_nome:
        raise ValueError("Estrutura inesperada no .mpx (sem os metadados do projeto).")
    meta = json.loads(z.read(meta_nome[0]).decode("utf-8-sig"))
    atual = meta.get("Worksheets", {}).get("ExtraInformation", {}).get("current_worksheet")
    historico = " ".join(str(h) for h in meta.get("History", []))
    decimal = "," if re.search(r"Levels[^;.\n]*\d,\d", historico) else (
        "." if re.search(r"Levels[^;\n]*\d\.\d", historico) else ",")
    planilhas = []
    for w in meta.get("Worksheets", {}).get("Items", []):
        try:
            d = ler(w["Uri"] + ".json")["Data"]
        except (KeyError, ValueError):
            continue
        colunas = {}
        for c in d.get("Columns", []):
            corpo = c.get("WorksheetVarBody") or {}
            nome = str(corpo.get("Name", "")).strip()
            vb = (corpo.get("VarData") or {}).get("VarDataBody") or {}
            if not nome:
                continue
            if vb.get("HasNumericData") and "NumericData" in vb:
                vals = [float("nan") if (v is None or abs(v) > AUSENTE_MINITAB) else float(v)
                        for v in vb["NumericData"]]
            elif vb.get("HasTextData"):
                vals = list(vb.get("TextData") or vb.get("StringData") or [])
            else:
                vals = []
            colunas[nome] = vals
        fatores = []
        try:
            des = d["DOE"]["Design"]
            fatores = [str(f.get("FullName") or f.get("Name")).strip()
                       for f in des["Factors"][:int(des["FactorCt"])]]
            fatores = [f for f in fatores if f in colunas]
        except (KeyError, TypeError, ValueError):
            pass
        if colunas:
            planilhas.append({"nome": w.get("Name") or d.get("PrivateTitle") or "Worksheet",
                              "atual": w.get("Id") == atual, "colunas": colunas, "fatores": fatores})
    if not planilhas:
        raise ValueError("Nao encontrei nenhuma planilha com dados nesse projeto.")
    return {"planilhas": planilhas, "decimal": decimal}


def ler_tabela(caminho):
    ext = os.path.splitext(caminho)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        t = pd.read_excel(caminho, dtype=str)
    else:
        t = pd.read_csv(caminho, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    t = t.loc[:, [c for c in t.columns if not str(c).startswith("Unnamed")]].fillna("")
    decimal = "," if t.apply(lambda c: c.astype(str).str.contains(r"\d,\d", regex=True)).to_numpy().any() else "."
    colunas = {}
    for c in t.columns:
        bruto = [str(v).strip() for v in t[c]]
        nums = [num(v) for v in bruto]
        if all(v == "" for v in bruto):
            colunas[str(c).strip()] = []
        elif all(v == "" or not math.isnan(n) for v, n in zip(bruto, nums)):
            colunas[str(c).strip()] = nums
        else:
            colunas[str(c).strip()] = bruto
    if not colunas:
        raise ValueError("A planilha esta vazia.")
    return {"planilhas": [{"nome": os.path.basename(caminho), "atual": True,
                           "colunas": colunas, "fatores": []}], "decimal": decimal}


def _numerica_cheia(v, n):
    return len(v) == n and all(isinstance(x, float) and not math.isnan(x) for x in v)


def opcoes_de_id(plan):
    col = plan["colunas"]
    n = max((len(v) for v in col.values()), default=0)
    cand = [c for c, v in col.items() if len(v) == n and len({id_ensaio(x) for x in v}) == n
            and not any(isinstance(x, float) and math.isnan(x) for x in v) and c not in plan["fatores"]]
    pref = [c for papel in ("padrao", "execucao") for c in cand if _chave(c) in PAPEIS[papel]]
    return pref + [c for c in cand if c not in pref] + [LINHA]


def montar_de_planilha(plan, id_col=None):
    col = plan["colunas"]
    n = max((len(v) for v in col.values()), default=0)
    if n < 2:
        raise ValueError("A planilha tem menos de 2 linhas de dados.")
    papel = {p: next((c for c in col if _chave(c) in nomes), None) for p, nomes in PAPEIS.items()}
    reserv = {c for c in papel.values() if c}
    opcoes = opcoes_de_id(plan)
    id_col = id_col if id_col in opcoes else opcoes[0]
    fatores = [f for f in plan["fatores"] if f in col and _numerica_cheia(col[f], n)]
    if not fatores:
        fatores = [c for c, v in col.items() if c not in reserv and c != id_col
                   and _numerica_cheia(v, n) and len(set(v)) >= 2]
    if len(fatores) > len(LETRAS):
        fatores = fatores[:len(LETRAS)]
    real = np.column_stack([col[f] for f in fatores]).astype(float) if fatores else np.zeros((n, 0))
    k = len(fatores)
    ids = [str(i + 1) for i in range(n)] if id_col == LINHA else [id_ensaio(v) for v in col[id_col]]
    exe = papel["execucao"]
    ordem = [int(v) for v in col[exe]] if exe and _numerica_cheia(col[exe], n) else list(range(1, n + 1))
    df = pd.DataFrame({"Linha": np.arange(1, n + 1), "Ensaio": ids, "Ordem": ordem})
    for j, f in enumerate(fatores):
        df[f] = real[:, j]
    numericas = [c for c, v in col.items() if _numerica_cheia(v, n)]
    vazias = [c for c, v in col.items()
              if len(v) == 0 or all(isinstance(x, float) and math.isnan(x) for x in v)]
    meta = {"k": k, "letras": list(LETRAS[:k]), "nomes": list(fatores), "id_col": id_col,
            "opcoes_id": opcoes, "vazias": vazias, "planilha": plan["nome"],
            "numericas": numericas, "colunas": {c: list(v) for c, v in col.items()}}
    return df, meta


def descrever_planejamento(df, meta):
    L = [f"Planilha '{meta['planilha']}': {len(df)} ensaios, {meta['k']} coluna(s) numerica(s) de fator.",
         ("Fatores: " + ", ".join(meta["nomes"]) + ".") if meta["nomes"] else "Nenhum fator numerico identificado.",
         f"Os tubos sao identificados pela coluna '{meta['id_col']}'."]
    if meta["vazias"]:
        L.append("Coluna(s) ainda vazia(s) no arquivo: " + ", ".join(meta["vazias"]) + ".")
    return "\n".join(L)


# ============================================================================
# CALIBRACAO (concentracao x resposta)
# ============================================================================
def calibrar(conc, resp):
    """Regressao linear y = a + b*x. Devolve coeficientes, R2, LOD, LOQ e previsao."""
    x = np.asarray(conc, float)
    y = np.asarray(resp, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    out = {"n": int(len(x)), "x": x, "y": y, "a": np.nan, "b": np.nan, "R2": np.nan,
           "Syx": np.nan, "LOD": np.nan, "LOQ": np.nan, "EP_b": np.nan, "EP_a": np.nan}
    if len(x) < 2 or np.ptp(x) == 0:
        return out
    b, a = np.polyfit(x, y, 1)
    prev = a + b * x
    res = y - prev
    sqt = float(((y - y.mean()) ** 2).sum())
    sqe = float((res ** 2).sum())
    out["a"], out["b"] = float(a), float(b)
    out["R2"] = float(1 - sqe / sqt) if sqt > 0 else np.nan
    if len(x) > 2:
        syx = math.sqrt(sqe / (len(x) - 2))
        sxx = float(((x - x.mean()) ** 2).sum())
        out["Syx"] = syx
        out["EP_b"] = syx / math.sqrt(sxx) if sxx > 0 else np.nan
        out["EP_a"] = syx * math.sqrt(1 / len(x) + x.mean() ** 2 / sxx) if sxx > 0 else np.nan
        if b != 0:
            out["LOD"] = 3.3 * syx / abs(b)
            out["LOQ"] = 10.0 * syx / abs(b)
    return out


def prever_conc(cal, y):
    if cal["n"] < 2 or not cal["b"] or math.isnan(cal["b"]) or cal["b"] == 0:
        return float("nan")
    if y is None or (isinstance(y, float) and math.isnan(y)):
        return float("nan")
    return (float(y) - cal["a"]) / cal["b"]


def texto_calibracao(cal, resposta):
    if cal["n"] < 2:
        return ("Curva de calibracao: faltam pontos. Digite a concentracao de pelo menos 2 ensaios "
                "(coluna 'Conc.' na aba 1) com respostas ja lidas.")
    L = [f"Curva de calibracao com {cal['n']} ponto(s): {resposta} = "
         f"{cal['a']:.5g} + {cal['b']:.5g} x concentracao."]
    L.append(f"R² = {cal['R2']:.4f}" + ("" if math.isnan(cal["Syx"]) else
                                        f"; S(y/x) = {cal['Syx']:.5g}"))
    if not math.isnan(cal["EP_b"]):
        L.append(f"Sensibilidade (inclinacao) = {cal['b']:.5g} ± {cal['EP_b']:.2g}.")
    if not math.isnan(cal["LOD"]):
        L.append(f"LOD estimado = {cal['LOD']:.4g}; LOQ estimado = {cal['LOQ']:.4g} "
                 "(3,3 e 10 x S(y/x) / inclinacao).")
    if not math.isnan(cal["R2"]) and cal["R2"] < 0.98:
        L.append("AVISO: R² abaixo de 0,98 - confira as ROIs, o branco e as fotos dos pontos fora da reta.")
    return "\n".join(L)


def grafico_calibracao(fig, cal, pontos_desc, resposta):
    fig.clear()
    fig.set_facecolor(FUNDO)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=SEC)
    if cal["n"] >= 1:
        ax.scatter(cal["x"], cal["y"], s=46, color=AZUL, edgecolor=FUNDO, zorder=3, label="padroes")
    if cal["n"] >= 2 and not math.isnan(cal["b"]):
        gx = np.linspace(float(np.min(cal["x"])), float(np.max(cal["x"])), 50)
        ax.plot(gx, cal["a"] + cal["b"] * gx, color=TINTA, lw=1.3, ls="--",
                label=f"y = {cal['a']:.4g} + {cal['b']:.4g}x  (R² = {cal['R2']:.4f})")
    if pontos_desc:
        xs = [p[0] for p in pontos_desc]
        ys = [p[1] for p in pontos_desc]
        ax.scatter(xs, ys, s=40, marker="^", color=LARANJA, edgecolor=FUNDO, zorder=3,
                   label="amostras (conc. prevista)")
    ax.set_xlabel("Concentracao", color=SEC)
    ax.set_ylabel(resposta, color=SEC)
    ax.set_title("Curva de calibracao", loc="left", fontweight="bold", color=TINTA)
    ax.grid(color="#e8e7e2", lw=0.8)
    if cal["n"] or pontos_desc:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()


# ============================================================================
# INTERFACE GRAFICA
# ============================================================================
def rodar_gui():
    import re
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from PIL import Image, ImageTk

    APP = "DoE RGB Studio"
    EH_BRANCO = re.compile(r"^B\d+$")

    class Fluxo(ttk.Frame):
        """Barra de ferramentas que redistribui os grupos em linhas conforme a largura."""

        def __init__(self, master, pad=6, **kw):
            super().__init__(master, **kw)
            self._itens, self._pad, self._larg = [], pad, 0
            self.bind("<Configure>", self._mudou)

        def grupo(self, titulo):
            g = ttk.Labelframe(self, text=titulo, padding=(6, 2, 6, 4))
            self._itens.append(g)
            return g

        def item(self, w):
            self._itens.append(w)
            return w

        def _mudou(self, ev):
            if abs(ev.width - self._larg) > 6:
                self._larg = ev.width
                self.after_idle(lambda: self.refluir(ev.width))

        def refluir(self, larg=None):
            larg = larg or self.winfo_width() or 10000
            x = y = alt = 0
            for w in self._itens:
                lw, lh = w.winfo_reqwidth(), w.winfo_reqheight()
                if x and x + lw > larg:
                    x, y, alt = 0, y + alt + self._pad, 0
                w.place(x=x, y=y, width=lw, height=lh)
                x += lw + self._pad
                alt = max(alt, lh)
            self.configure(height=y + alt + 2)

    def editar_celula(tree, ev, editaveis, ao_salvar):
        item, col = tree.identify_row(ev.y), tree.identify_column(ev.x)
        if not item or not col:
            return
        nome_col = tree["columns"][int(col[1:]) - 1]
        if nome_col not in editaveis:
            return
        caixa = tree.bbox(item, col)
        if not caixa:
            return
        x, y, w, h = caixa
        ent = ttk.Entry(tree)
        ent.insert(0, tree.set(item, nome_col))
        ent.select_range(0, "end")
        ent.place(x=x, y=y, width=w, height=h)
        ent.focus_set()
        estado = {"feito": False}

        def fechar(salvar):
            if estado["feito"]:
                return
            estado["feito"] = True
            valor = ent.get()
            ent.destroy()
            if salvar:
                ao_salvar(item, nome_col, valor)

        ent.bind("<Return>", lambda _e: fechar(True))
        ent.bind("<KP_Enter>", lambda _e: fechar(True))
        ent.bind("<FocusOut>", lambda _e: fechar(True))
        ent.bind("<Escape>", lambda _e: fechar(False))

    def fmt(v, casas=3):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{casas}f}"

    class App(tk.Tk):
        COLS_FOTOS = ["Foto", "IDs no nome", "ROIs", "Branco", "Situacao"]
        COLS_LEIT = ["ID", "Conc.", "R", "G", "B", "Bruto", "Resposta"]

        def __init__(self):
            super().__init__()
            self.title(APP + " - leitura colorimetrica por foto")
            self.geometry("1480x920")
            self.fonte, self.caminho_fonte = None, ""
            self.df, self.meta = None, None
            self.manual, self.conc = {}, {}
            self.arquivos, self.idx, self.atual = [], 0, None
            self.imagens, self.rois, self.brancos = {}, {}, {}
            self.hist = {}
            self.conf = set()
            self._cache, self._linhas_rois = {}, []
            self.fator, self.foto, self._chave_foto = 1.0, None, None
            self._arrasto, self.sel = None, None
            self.sinal, self.cal = None, calibrar([], [])

            self.lado = tk.IntVar(value=LADO_PADRAO)
            self.resp = tk.StringVar(value="A_R")
            self.prox = tk.StringVar(value="")
            self.usar_br = tk.BooleanVar(value=False)
            self.marcar_b = tk.BooleanVar(value=False)
            self.descontar = tk.BooleanVar(value=True)
            self.auto_num = tk.BooleanVar(value=True)
            self.n_inicial = tk.IntVar(value=1)
            self.sep = tk.StringVar(value="virgula")
            self.casas = tk.IntVar(value=4)
            self.planilha = tk.StringVar()
            self.id_col = tk.StringVar()
            self.col_conc = tk.StringVar(value=SEM_CONC)

            abas = ttk.Notebook(self)
            abas.pack(fill="both", expand=True)
            self.aba1, self.aba2, self.aba3 = (ttk.Frame(abas) for _ in range(3))
            abas.add(self.aba1, text="1. Planejamento")
            abas.add(self.aba2, text="2. Fotos e ROIs")
            abas.add(self.aba3, text="3. Sinal e calibracao")
            self.status = ttk.Label(self, anchor="w", padding=4,
                                    text="Abra o planejamento (.mpx, CSV ou XLSX) ou va direto para a aba 2.")
            self.status.pack(fill="x")
            self._montar_aba1()
            self._montar_aba2()
            self._montar_aba3()
            self.alternar_branco_reagente()

        def avisar(self, texto):
            self.status.config(text=texto)

        def eh_branco(self, i):
            return bool(self.usar_br.get() and EH_BRANCO.match(str(i)))

        # ============================================================ ABA 1
        def _montar_aba1(self):
            b = Fluxo(self.aba1, padding=(8, 6, 8, 2))
            b.pack(fill="x")
            g = b.grupo("Arquivo")
            ttk.Button(g, text="Minitab (.mpx)...", command=self.abrir_mpx).pack(side="left")
            ttk.Button(g, text="Planilha (CSV/XLSX)...", command=self.abrir_tabela).pack(side="left", padx=4)
            g = b.grupo("Sessao")
            ttk.Button(g, text="Abrir...", command=self.abrir_sessao).pack(side="left")
            ttk.Button(g, text="Salvar...", command=self.salvar_sessao).pack(side="left", padx=4)
            g = b.grupo("Colunas")
            ttk.Label(g, text="Planilha:").pack(side="left")
            self.cb_plan = ttk.Combobox(g, textvariable=self.planilha, values=[], state="readonly", width=22)
            self.cb_plan.pack(side="left", padx=(4, 10))
            self.cb_plan.bind("<<ComboboxSelected>>", lambda _e: self.montar())
            ttk.Label(g, text="ID dos tubos:").pack(side="left")
            self.cb_id = ttk.Combobox(g, textvariable=self.id_col, values=[], state="readonly", width=20)
            self.cb_id.pack(side="left", padx=(4, 10))
            self.cb_id.bind("<<ComboboxSelected>>", lambda _e: self.montar(manter_id=True))
            ttk.Label(g, text="Concentracao:").pack(side="left")
            self.cb_conc = ttk.Combobox(g, textvariable=self.col_conc, values=[SEM_CONC],
                                        state="readonly", width=20)
            self.cb_conc.pack(side="left", padx=4)
            self.cb_conc.bind("<<ComboboxSelected>>", lambda _e: self.importar_conc())
            g = b.grupo("Respostas sem foto")
            ttk.Button(g, text="Preencher em lote...", command=self.preencher_manuais).pack(side="left")
            ttk.Button(g, text="Limpar manuais", command=self.limpar_manuais).pack(side="left", padx=4)
            b.refluir(1200)

            self.descr = tk.Text(self.aba1, wrap="word", height=5, font=("TkDefaultFont", 9))
            self.descr.pack(fill="x", padx=8, pady=8)
            ttk.Label(self.aba1, padding=(8, 0), text=(
                "Duplo clique para editar 'Conc.' (concentracao do padrao) e 'Manual' (resposta digitada, "
                "tem prioridade sobre a foto). 'Foto' = sinal das fotos confirmadas na aba 2.")).pack(anchor="w")
            quadro = ttk.Frame(self.aba1, padding=8)
            quadro.pack(fill="both", expand=True)
            self.tab_mz = ttk.Treeview(quadro, columns=["Ensaio"], show="headings")
            sy = ttk.Scrollbar(quadro, orient="vertical", command=self.tab_mz.yview)
            sx = ttk.Scrollbar(quadro, orient="horizontal", command=self.tab_mz.xview)
            self.tab_mz.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
            sy.pack(side="right", fill="y")
            sx.pack(side="bottom", fill="x")
            self.tab_mz.pack(fill="both", expand=True)
            self.tab_mz.bind("<Double-1>", lambda ev: editar_celula(
                self.tab_mz, ev, ["Manual", "Conc."], self.salvar_celula))

        def abrir_mpx(self):
            arq = filedialog.askopenfilename(title="Projeto do Minitab",
                                             filetypes=[("Projeto do Minitab", "*.mpx"), ("Todos", "*.*")])
            if arq:
                self.carregar_fonte(arq, ler_mpx)

        def abrir_tabela(self):
            arq = filedialog.askopenfilename(
                title="Planilha do planejamento",
                filetypes=[("Planilhas", "*.csv *.xlsx *.xls *.txt"), ("Todos", "*.*")])
            if arq:
                self.carregar_fonte(arq, ler_tabela)

        def carregar_fonte(self, arq, leitor):
            try:
                fonte = leitor(arq)
            except Exception as e:
                messagebox.showerror(APP, f"Nao foi possivel ler o arquivo:\n{e}")
                return False
            if self.df is not None and (self.manual or self.conc) and not messagebox.askyesno(
                    APP, "Abrir outro planejamento apaga as respostas manuais e as concentracoes. Continuar?"):
                return False
            self.fonte, self.caminho_fonte = fonte, arq
            nomes = [p["nome"] for p in fonte["planilhas"]]
            self.cb_plan.configure(values=nomes)
            self.planilha.set(next((p["nome"] for p in fonte["planilhas"] if p["atual"]), nomes[-1]))
            self.sep.set("virgula" if fonte["decimal"] == "," else "ponto")
            return self.montar()

        def montar(self, manter_id=False, manual=None, conc=None):
            if not self.fonte:
                return False
            plan = next((p for p in self.fonte["planilhas"] if p["nome"] == self.planilha.get()),
                        self.fonte["planilhas"][0])
            try:
                df, meta = montar_de_planilha(plan, self.id_col.get() if manter_id else None)
            except Exception as e:
                messagebox.showerror(APP, f"Nao foi possivel interpretar a planilha '{plan['nome']}':\n{e}")
                return False
            self.cb_id.configure(values=meta["opcoes_id"])
            self.id_col.set(meta["id_col"])
            self.cb_conc.configure(values=[SEM_CONC] + meta["numericas"])
            self.df, self.meta = df, meta
            self.manual = dict(manual or {})
            self.conc = dict(conc or {})
            if conc is None:
                self.col_conc.set(SEM_CONC)
            self.sinal = None
            self._cache = {}
            self.descr.delete("1.0", "end")
            self.descr.insert("1.0", f"Arquivo: {self.caminho_fonte}\n" + descrever_planejamento(df, meta))
            self.atualizar_proximo()
            self.atualizar_matriz()
            self.atualizar_fotos()
            self.avisar(f"Planejamento carregado: {len(df)} ensaios. Nomeie as fotos com os IDs da coluna "
                        f"'{meta['id_col']}' (ex.: 9-10-11.jpeg) ou use a numeracao automatica na aba 2.")
            return True

        def importar_conc(self):
            if self.df is None:
                return
            c = self.col_conc.get()
            if c == SEM_CONC or c not in self.meta["colunas"]:
                return
            vals = self.meta["colunas"][c]
            for e, v in zip(self.df["Ensaio"], vals):
                if isinstance(v, float) and math.isnan(v):
                    continue
                self.conc[e] = str(v).replace(".", ",") if self.sep.get() == "virgula" else str(v)
            self.atualizar_matriz()
            self.avisar(f"Concentracoes importadas da coluna '{c}'.")

        def ids_plano(self):
            return [] if self.df is None else list(self.df["Ensaio"])

        def atualizar_matriz(self):
            if self.df is None:
                return
            resp = self.respostas()
            cols = ["Linha", "Ensaio", "Ordem"] + self.meta["nomes"] + ["Conc.", "Manual", "Foto", "Usada"]
            self.tab_mz.configure(columns=cols)
            for c in cols:
                self.tab_mz.heading(c, text=c)
                self.tab_mz.column(c, width=62 if c in ("Linha", "Ensaio", "Ordem") else 110, anchor="center")
            self.tab_mz.delete(*self.tab_mz.get_children())
            for _, r in self.df.iterrows():
                e = r["Ensaio"]
                self.tab_mz.insert("", "end", iid=e, values=(
                    [int(r["Linha"]), e, int(r["Ordem"])] + [f"{r[n]:g}" for n in self.meta["nomes"]]
                    + [self.conc.get(e, ""), self.manual.get(e, ""),
                       fmt(resp["foto"].get(e)), fmt(resp["usada"].get(e))]))

        def salvar_celula(self, item, coluna, valor):
            valor = valor.strip()
            if valor and math.isnan(num(valor)):
                messagebox.showerror(APP, "Digite um numero (ex.: 0,022) ou deixe em branco.")
                return
            alvo = self.conc if coluna == "Conc." else self.manual
            if valor:
                alvo[item] = valor
            else:
                alvo.pop(item, None)
            self.sinal = None
            self.atualizar_matriz()
            self.atualizar_proximo()

        def preencher_manuais(self):
            if self.df is None:
                return
            resp = self.respostas()
            faltam = [e for e in self.ids_plano() if e not in resp["usada"]]
            if not faltam:
                messagebox.showinfo(APP, "Todos os ensaios ja tem resposta.")
                return
            sugest = 0.0 if (not resp["desconto"] or math.isnan(resp["branco_medio"])) else resp["branco_medio"]
            v = simpledialog.askstring(APP, (
                f"{len(faltam)} ensaio(s) sem leitura: {', '.join(faltam)}.\n\n"
                "Valor a atribuir a todos eles (ex.: ensaios sem cor):"),
                initialvalue=f"{sugest:.3f}", parent=self)
            if v is None or math.isnan(num(v)):
                return
            for e in faltam:
                self.manual[e] = v.strip()
            self.sinal = None
            self.atualizar_matriz()
            self.atualizar_proximo()

        def limpar_manuais(self):
            if self.manual and messagebox.askyesno(APP, "Apagar todas as respostas manuais?"):
                self.manual = {}
                self.sinal = None
                self.atualizar_matriz()

        # ============================================================ ABA 2
        def _montar_aba2(self):
            barra = Fluxo(self.aba2, padding=(6, 4, 6, 2))
            barra.pack(fill="x")
            self.barra2 = barra

            g = barra.grupo("Fotos")
            ttk.Button(g, text="Abrir...", width=8, command=self.abrir).pack(side="left")
            ttk.Button(g, text="<", width=3, command=lambda: self.trocar(-1)).pack(side="left", padx=(6, 0))
            ttk.Button(g, text=">", width=3, command=lambda: self.trocar(1)).pack(side="left")
            ttk.Button(g, text="Tirar da lista", command=self.tirar_foto).pack(side="left", padx=(6, 0))

            g = barra.grupo("ROIs")
            ttk.Button(g, text="Detectar", command=lambda: self.detectar(self.atual, forcar=True)).pack(side="left")
            ttk.Button(g, text="Apagar (Del)", command=self.remover_sel).pack(side="left", padx=4)
            ttk.Button(g, text="Limpar foto", command=self.limpar).pack(side="left")
            ttk.Label(g, text="lado:").pack(side="left", padx=(8, 2))
            ttk.Spinbox(g, from_=6, to=200, textvariable=self.lado, width=4,
                        command=self.recalcular).pack(side="left")

            g = barra.grupo("Desfazer")
            ttk.Button(g, text="↶ Ctrl+Z", command=self.desfazer).pack(side="left")
            ttk.Button(g, text="↷ Ctrl+Y", command=self.refazer).pack(side="left", padx=4)

            g = barra.grupo("Numeracao dos IDs")
            ttk.Checkbutton(g, text="crescente automatica", variable=self.auto_num,
                            command=self.recalcular).pack(side="left")
            ttk.Label(g, text="de:").pack(side="left", padx=(8, 2))
            ttk.Spinbox(g, from_=0, to=9999, textvariable=self.n_inicial, width=4).pack(side="left")
            ttk.Button(g, text="Renumerar tudo", command=self.renumerar_tudo).pack(side="left", padx=4)
            ttk.Label(g, text="proximo ID:").pack(side="left", padx=(8, 2))
            self.cb_prox = ttk.Combobox(g, textvariable=self.prox, values=[], width=7)
            self.cb_prox.pack(side="left")

            g = barra.grupo("Resposta")
            ttk.Label(g, text="canal:").pack(side="left")
            cb = ttk.Combobox(g, textvariable=self.resp, values=RESPOSTAS, width=8, state="readonly")
            cb.pack(side="left", padx=(2, 0))
            cb.bind("<<ComboboxSelected>>", lambda _e: self.recalcular())

            g = barra.grupo("Branco do reagente")
            ttk.Checkbutton(g, text="usar", variable=self.usar_br,
                            command=self.alternar_branco_reagente).pack(side="left")
            self.ck_marcar = ttk.Checkbutton(g, text="novo clique = B1, B2...", variable=self.marcar_b)
            self.ck_marcar.pack(side="left", padx=4)
            self.ck_desc = ttk.Checkbutton(g, text="descontar media", variable=self.descontar,
                                           command=self.recalcular)
            self.ck_desc.pack(side="left")

            g = barra.grupo("Conferencia")
            ttk.Button(g, text="Confirmar esta foto", command=self.confirmar).pack(side="left")
            ttk.Button(g, text="Confirmar todas", command=self.confirmar_todas).pack(side="left", padx=4)
            barra.refluir(1400)

            corpo = ttk.Panedwindow(self.aba2, orient="horizontal")
            corpo.pack(fill="both", expand=True)
            self.canvas = tk.Canvas(corpo, bg="#333333", cursor="crosshair", highlightthickness=0,
                                    takefocus=True)
            corpo.add(self.canvas, weight=3)
            dir_ = ttk.Frame(corpo)
            corpo.add(dir_, weight=2)
            ttk.Label(dir_, padding=4, justify="left", text=(
                "ROI retangular com vertices editaveis:\n"
                "  - arrastar dentro: move a ROI        - arrastar um quadradinho: move o vertice\n"
                "  - arrastar no vazio: cria uma ROI    - roda do mouse: aumenta/diminui a ROI\n"
                "  - Ctrl+clique numa aresta: adiciona vertice   - Ctrl+clique num vertice: remove\n"
                "  - Delete: apaga a ROI selecionada    - Ctrl+Z / Ctrl+Y: desfazer / refazer\n"
                "  - clique direito: branco de referencia (fundo)\n"
                "  - duplo clique na tabela: troca o ID; na coluna 'Conc.', a concentracao do padrao"
            )).pack(anchor="w")
            self.tab_fotos = ttk.Treeview(dir_, columns=self.COLS_FOTOS, show="headings",
                                          selectmode="browse", height=9)
            for c, w in zip(self.COLS_FOTOS, (150, 110, 45, 70, 170)):
                self.tab_fotos.heading(c, text=c)
                self.tab_fotos.column(c, width=w, anchor="center")
            self.tab_fotos.pack(fill="x")
            self.tab_fotos.bind("<<TreeviewSelect>>", self.ao_escolher_foto)
            ttk.Label(dir_, padding=(4, 8, 4, 2), text="ROIs da foto atual:").pack(anchor="w")
            self.tab_leit = ttk.Treeview(dir_, columns=self.COLS_LEIT, show="headings", selectmode="browse")
            for c in self.COLS_LEIT:
                self.tab_leit.heading(c, text=c)
                self.tab_leit.column(c, width=66, anchor="center")
            self.tab_leit.pack(fill="both", expand=True)
            self.tab_leit.bind("<Double-1>", self.duplo_clique_roi)
            self.tab_leit.bind("<<TreeviewSelect>>", self.ao_escolher_roi)

            self.canvas.bind("<ButtonPress-1>", self.pressionar)
            self.canvas.bind("<B1-Motion>", self.arrastar)
            self.canvas.bind("<ButtonRelease-1>", self.soltar)
            self.canvas.bind("<Button-3>", self.clique_branco)
            self.canvas.bind("<Button-2>", self.clique_branco)
            self.canvas.bind("<MouseWheel>", lambda ev: self.roda(ev, 1 if ev.delta > 0 else -1))
            self.canvas.bind("<Button-4>", lambda ev: self.roda(ev, 1))
            self.canvas.bind("<Button-5>", lambda ev: self.roda(ev, -1))
            self.canvas.bind("<Configure>", lambda _e: self.mostrar_foto())
            self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set())
            for seq, fn in (("<Delete>", self.remover_sel), ("<BackSpace>", self.remover_sel),
                            ("<Control-z>", self.desfazer), ("<Control-Z>", self.desfazer),
                            ("<Control-y>", self.refazer), ("<Control-Y>", self.refazer),
                            ("<Control-Shift-Z>", self.refazer)):
                self.canvas.bind(seq, lambda _e, f=fn: (f(), "break")[1])

        def alternar_branco_reagente(self):
            on = bool(self.usar_br.get())
            estado = "normal" if on else "disabled"
            self.ck_marcar.configure(state=estado)
            self.ck_desc.configure(state=estado)
            if not on:
                self.marcar_b.set(False)
            self.recalcular()

        # ------------------------------------------------- historico (undo/redo)
        def _estado(self, arq):
            return {"rois": json.loads(json.dumps(self.rois.get(arq, []))),
                    "branco": json.loads(json.dumps(self.brancos.get(arq)))}

        def marcar_hist(self, arq=None):
            arq = arq or self.atual
            if not arq:
                return
            h = self.hist.setdefault(arq, {"und": [], "red": []})
            h["und"].append(self._estado(arq))
            del h["und"][:-80]
            h["red"].clear()

        def _aplicar(self, arq, est):
            self.rois[arq] = est["rois"]
            if est["branco"] is None:
                self.brancos.pop(arq, None)
            else:
                self.brancos[arq] = est["branco"]
            self.sel = None

        def desfazer(self):
            h = self.hist.get(self.atual)
            if not self.atual or not h or not h["und"]:
                self.avisar("Nada para desfazer nesta foto.")
                return
            h["red"].append(self._estado(self.atual))
            self._aplicar(self.atual, h["und"].pop())
            self.mexeu()
            self.recalcular()

        def refazer(self):
            h = self.hist.get(self.atual)
            if not self.atual or not h or not h["red"]:
                self.avisar("Nada para refazer nesta foto.")
                return
            h["und"].append(self._estado(self.atual))
            self._aplicar(self.atual, h["red"].pop())
            self.mexeu()
            self.recalcular()

        # ----------------------------------------------------- fotos e deteccao
        def abrir(self):
            arqs = filedialog.askopenfilenames(
                title="Selecione as fotos",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")])
            if not arqs:
                return
            novos = [a for a in arqs if a not in self.arquivos]
            ruins = []
            for a in novos:
                try:
                    self.imagem(a)
                except IOError:
                    ruins.append(os.path.basename(a))
                    continue
                self.arquivos.append(a)
                self.rois.setdefault(a, [])
                self.detectar(a, atualizar=False)
            if self.arquivos:
                self.idx = self.arquivos.index(arqs[0]) if arqs[0] in self.arquivos else 0
                self.atual = self.arquivos[self.idx]
            if self.auto_num.get():
                self.renumerar_tudo(silencioso=True)
            self.recalcular()
            if ruins:
                messagebox.showwarning(APP, "Nao consegui abrir: " + ", ".join(ruins))

        def imagem(self, arq):
            if arq not in self.imagens:
                self.imagens[arq] = ler_imagem(arq)
            return self.imagens[arq]

        def detectar(self, arq, forcar=False, atualizar=True):
            if not arq:
                return
            if forcar and self.rois.get(arq) and not messagebox.askyesno(
                    APP, "Detectar de novo apaga as ROIs desta foto. Continuar?"):
                return
            if forcar:
                self.marcar_hist(arq)
            img = self.imagem(arq)
            n = len(self.partes_validas(arq)) or None
            try:
                caixas = detectar_rois(img, n, self.lado.get())
            except Exception:
                caixas = []
            self.rois[arq] = [{"id": None, "pts": c} for c in caixas]
            if self.auto_num.get():
                self.renumerar_tudo(silencioso=True)
            elif not self.partes_validas(arq):
                for p in self.rois[arq]:
                    p["id"] = "?"
            if forcar or arq not in self.brancos or self.brancos[arq].get("auto"):
                xy = branco_automatico(img, [centro_roi(p) for p in self.rois[arq]])
                rgb = rgb_do_branco(img, *xy) if xy else None
                if rgb:
                    self.brancos[arq] = {"rgb": rgb, "xy": list(xy), "auto": True}
            self.conf.discard(arq)
            self.sel = None
            if atualizar:
                self.recalcular()

        def trocar(self, passo):
            if self.arquivos:
                self.idx = (self.idx + passo) % len(self.arquivos)
                self.atual = self.arquivos[self.idx]
                self.sel = None
                self.atualizar_fotos()

        def ao_escolher_foto(self, _ev=None):
            sel = self.tab_fotos.selection()
            if not sel:
                return
            try:
                i = int(sel[0])
            except ValueError:
                return
            if 0 <= i < len(self.arquivos) and self.arquivos[i] != self.atual:
                self.idx, self.atual, self.sel = i, self.arquivos[i], None
                self.mostrar_foto()
                self.atualizar_leituras()

        def ao_escolher_roi(self, _ev=None):
            sel = self.tab_leit.selection()
            if sel and self._linhas_rois:
                i = self.tab_leit.index(sel[0])
                if 0 <= i < len(self._linhas_rois):
                    self.sel = self._linhas_rois[i]
                    self.desenhar_rois()

        def tirar_foto(self):
            if not self.atual:
                return
            a = self.atual
            self.arquivos.remove(a)
            for d in (self.rois, self.brancos, self.imagens, self.hist):
                d.pop(a, None)
            self.conf.discard(a)
            self.idx = min(self.idx, max(0, len(self.arquivos) - 1))
            self.atual = self.arquivos[self.idx] if self.arquivos else None
            self.sel = None
            self.recalcular()

        # ------------------------------------------------------ IDs das ROIs
        def partes_validas(self, arq):
            if self.auto_num.get():
                return []
            partes = partes_do_arquivo(arq)
            if not partes or len(set(partes)) != len(partes):
                return []
            plano = set(self.ids_plano())
            if plano:
                return partes if all(p in plano or self.eh_branco(p) for p in partes) else []
            return partes if all(not math.isnan(num(p)) or self.eh_branco(p) for p in partes) else []

        def ordenadas(self, arq):
            return sorted(self.rois.get(arq, []), key=lambda p: centro_roi(p)[0])

        def ids_da_foto(self, arq):
            rois = self.ordenadas(arq)
            partes = self.partes_validas(arq)
            autos = [p for p in rois if not p.get("id")]
            usados = {p["id"] for p in rois if p.get("id")}
            livres = [x for x in partes if x not in usados]
            prov = "?" if len(autos) < len(livres) else ""
            mapa = {id(p): n + prov for p, n in zip(autos[:len(livres)], livres)}
            return rois, [p.get("id") or mapa.get(id(p), "?") for p in rois]

        def renumerar_tudo(self, silencioso=False):
            """Numera todas as ROIs em ordem crescente: fotos na ordem da lista, ROIs da esquerda para a direita."""
            if not silencioso:
                for arq in self.arquivos:
                    self.marcar_hist(arq)
            n = int(self.n_inicial.get())
            for arq in self.arquivos:
                for p in self.ordenadas(arq):
                    if self.eh_branco(p.get("id")):
                        continue
                    p["id"] = str(n)
                    n += 1
            if not silencioso:
                self.conf.clear()
                self.recalcular()
                self.avisar(f"ROIs renumeradas de {self.n_inicial.get()} a {n - 1}.")

        def proximo_numero(self):
            usados = []
            for arq in self.arquivos:
                for p in self.rois.get(arq, []):
                    v = num(p.get("id"))
                    if not math.isnan(v):
                        usados.append(int(v))
            return str(max(usados) + 1 if usados else int(self.n_inicial.get()))

        def situacao(self, arq):
            rois, ids = self.ids_da_foto(arq)
            partes = self.partes_validas(arq)
            if not rois:
                return "sem ROI: desenhe nos tubos"
            if any(i.endswith("?") for i in ids):
                falt = len(partes) - len(rois)
                if partes and falt > 0:
                    return f"faltam {falt} tubo(s)"
                if partes and falt < 0:
                    return f"{-falt} ROI(s) a mais: remova"
                return "defina o ID das ROIs '?'"
            if len(set(ids)) != len(ids):
                return "IDs repetidos nesta foto"
            if arq not in self.brancos:
                return "sem branco: clique direito"
            return "CONFIRMADA" if arq in self.conf else "a conferir"

        def todos_ids(self, so_confirmadas=False):
            lidos = set()
            for arq in self.arquivos:
                if not so_confirmadas or arq in self.conf:
                    lidos.update(i for i in self.ids_da_foto(arq)[1] if not i.endswith("?"))
            return lidos

        def atualizar_proximo(self):
            plano = self.ids_plano()
            self.cb_prox.configure(values=plano)
            feitos = self.todos_ids() | set(self.manual)
            pend = [e for e in plano if e not in feitos]
            if id_ensaio(self.prox.get()) not in pend:
                self.prox.set(pend[0] if pend else "")

        def confirmar(self, arq=None, silencioso=False):
            arq = arq or self.atual
            if not arq:
                return False
            sit = self.situacao(arq)
            if sit not in ("a conferir", "CONFIRMADA"):
                if not silencioso:
                    messagebox.showwarning(APP, f"{os.path.basename(arq)} ainda nao pode ser confirmada: {sit}.")
                return False
            self.conf.add(arq)
            if not silencioso:
                pend = [a for a in self.arquivos if a not in self.conf]
                if pend:
                    self.atual = pend[0]
                    self.idx = self.arquivos.index(self.atual)
                    self.sel = None
                self.recalcular()
            return True

        def confirmar_todas(self):
            ok = sum(self.confirmar(a, silencioso=True) for a in self.arquivos)
            self.recalcular()
            falt = len(self.arquivos) - ok
            if falt:
                messagebox.showwarning(APP, f"{ok} foto(s) confirmada(s). {falt} ainda precisam de ajuste "
                                            "(veja a coluna 'Situacao').")

        # ------------------------------------------------------- mouse no canvas
        def _xy(self, ev):
            return ev.x / self.fator, ev.y / self.fator

        def _tol(self):
            return max(7.0 / max(self.fator, 1e-6), 3.0)

        def vertice_em(self, x, y):
            tol = self._tol()
            melhor, dmin = None, 1e18
            for p in self.rois.get(self.atual, []):
                for i, (vx, vy) in enumerate(p["pts"]):
                    d = math.hypot(vx - x, vy - y)
                    if d < tol and d < dmin:
                        melhor, dmin = (p, i), d
            return melhor

        def roi_em(self, x, y):
            for p in reversed(self.rois.get(self.atual, [])):
                if dentro(p, x, y):
                    return p
            return None

        def mexeu(self):
            self.conf.discard(self.atual)
            self.sinal = None

        def pressionar(self, ev):
            if not self.atual:
                return
            self.canvas.focus_set()
            x, y = self._xy(ev)
            ctrl = bool(ev.state & 0x0004)
            v = self.vertice_em(x, y)
            if ctrl:
                if v:
                    p, i = v
                    if len(p["pts"]) > 3:
                        self.marcar_hist()
                        p["pts"].pop(i)
                        self.sel = p
                        self.mexeu()
                        self.recalcular()
                    else:
                        self.avisar("A ROI precisa de pelo menos 3 vertices.")
                    return
                alvo = self.roi_em(x, y) or self.sel
                if alvo:
                    i, d, q = aresta_mais_perto(alvo, x, y)
                    if d < max(self._tol() * 3, 10):
                        self.marcar_hist()
                        alvo["pts"].insert(i + 1, q)
                        self.sel = alvo
                        self.mexeu()
                        self.recalcular()
                return
            if v:
                self.marcar_hist()
                self.sel = v[0]
                self._arrasto = {"tipo": "vertice", "roi": v[0], "i": v[1], "moveu": False}
                self.desenhar_rois()
                return
            p = self.roi_em(x, y)
            if p is not None:
                self.marcar_hist()
                self.sel = p
                self._arrasto = {"tipo": "mover", "roi": p, "x0": x, "y0": y,
                                 "orig": [list(v) for v in p["pts"]], "moveu": False}
                self.desenhar_rois()
                return
            self._arrasto = {"tipo": "novo", "x0": x, "y0": y, "x1": x, "y1": y, "moveu": False}

        def arrastar(self, ev):
            if not self._arrasto or not self.atual:
                return
            x, y = self._xy(ev)
            img = self.imagem(self.atual)
            H, W = img.shape[:2]
            x = float(np.clip(x, 0, W - 1))
            y = float(np.clip(y, 0, H - 1))
            a = self._arrasto
            a["moveu"] = True
            if a["tipo"] == "vertice":
                a["roi"]["pts"][a["i"]] = [x, y]
            elif a["tipo"] == "mover":
                dx, dy = x - a["x0"], y - a["y0"]
                a["roi"]["pts"] = [[float(np.clip(vx + dx, 0, W - 1)), float(np.clip(vy + dy, 0, H - 1))]
                                   for vx, vy in a["orig"]]
            else:
                a["x1"], a["y1"] = x, y
            self.mexeu()
            self.desenhar_rois()

        def soltar(self, _ev):
            a, self._arrasto = self._arrasto, None
            if not a or not self.atual:
                return
            if a["tipo"] == "novo":
                esc = self.imagem(self.atual).shape[1] / LARGURA_REF
                lado = self.lado.get() * esc
                x0, y0, x1, y1 = a["x0"], a["y0"], a["x1"], a["y1"]
                if not a["moveu"] or abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
                    pts = retangulo(a["x0"], a["y0"], lado)
                else:
                    pts = [[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)],
                           [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]]
                self.marcar_hist()
                p = {"id": self.novo_id(), "pts": pts}
                self.rois.setdefault(self.atual, []).append(p)
                self.sel = p
                self.mexeu()
            self.recalcular()

        def novo_id(self):
            if self.usar_br.get() and self.marcar_b.get():
                nums = [int(i[1:]) for i in self.todos_ids() if EH_BRANCO.match(i)]
                return f"B{max(nums, default=0) + 1}"
            if self.auto_num.get():
                return self.proximo_numero()
            if self.partes_validas(self.atual):
                return None
            return id_ensaio(self.prox.get()) or "?"

        def roda(self, ev, passo):
            if not self.atual:
                return
            x, y = self._xy(ev)
            p = self.roi_em(x, y) or (self.vertice_em(x, y) or (None,))[0] or self.sel
            if p is None:
                return
            cx, cy = centro_roi(p)
            f = 1.0 + 0.08 * passo
            novo = [[cx + (vx - cx) * f, cy + (vy - cy) * f] for vx, vy in p["pts"]]
            if area_roi({"pts": novo}) < 6:
                return
            self.marcar_hist()
            p["pts"] = novo
            self.sel = p
            self.mexeu()
            self.recalcular()

        def clique_branco(self, ev):
            if not self.atual:
                return
            x, y = self._xy(ev)
            rgb = rgb_do_branco(self.imagem(self.atual), x, y)
            if rgb:
                self.marcar_hist()
                self.brancos[self.atual] = {"rgb": rgb, "xy": [x, y], "auto": False}
                self.mexeu()
                self.recalcular()

        def remover_sel(self):
            alvo = self.sel
            if alvo is None:
                sel = self.tab_leit.selection()
                if sel and self._linhas_rois:
                    alvo = self._linhas_rois[self.tab_leit.index(sel[0])]
            if alvo is None or not self.atual:
                self.avisar("Selecione uma ROI (clique nela) antes de apagar.")
                return
            self.marcar_hist()
            self.rois[self.atual] = [p for p in self.rois[self.atual] if p is not alvo]
            self.sel = None
            self.mexeu()
            self.recalcular()

        def limpar(self):
            if self.atual:
                self.marcar_hist()
                self.rois[self.atual] = []
                self.brancos.pop(self.atual, None)
                self.sel = None
                self.mexeu()
                self.recalcular()

        def duplo_clique_roi(self, ev):
            col = self.tab_leit.identify_column(ev.x)
            i = int(col[1:]) - 1 if col and col.startswith("#") else -1
            if 0 <= i < len(self.COLS_LEIT) and self.COLS_LEIT[i] == "Conc.":
                editar_celula(self.tab_leit, ev, ["Conc."], self.salvar_conc_roi)
            else:
                self.renomear(ev)

        def salvar_conc_roi(self, item, _coluna, valor):
            valor = valor.strip()
            if valor and math.isnan(num(valor)):
                messagebox.showerror(APP, "Digite um numero (ex.: 0,5) ou deixe em branco.")
                return
            if not self.atual:
                return
            i = self.tab_leit.index(item)
            ids = self.ids_da_foto(self.atual)[1]
            if not 0 <= i < len(ids):
                return
            e = ids[i].rstrip("?")
            if valor:
                self.conc[e] = valor
            else:
                self.conc.pop(e, None)
            self.sinal = None
            self.atualizar_leituras()
            self.atualizar_matriz()

        def renomear(self, ev):
            item = self.tab_leit.identify_row(ev.y)
            if not item or not self.atual:
                return
            i = self.tab_leit.index(item)
            rois, ids = self.ids_da_foto(self.atual)
            dica = "ID desta ROI" + (" (ou B1, B2... para branco de reagente):" if self.usar_br.get() else ":")
            novo = simpledialog.askstring(APP, dica, initialvalue=ids[i].rstrip("?"), parent=self)
            if novo and novo.strip():
                self.marcar_hist()
                rois[i]["id"] = id_ensaio(novo)
                self.mexeu()
                self.recalcular()

        # ------------------------------------------------------------ medidas
        def recalcular(self):
            self._cache = {}
            self.sinal = None
            self.atualizar_fotos()
            self.atualizar_proximo()
            self.atualizar_matriz()

        def medir(self, arq):
            rois, ids = self.ids_da_foto(arq)
            if not rois:
                return [], None
            br = self.brancos.get(arq)
            chave = json.dumps([arq, [[[round(v, 1) for v in pt] for pt in p["pts"]] for p in rois],
                                ids, br], default=str)
            if chave not in self._cache:
                self._cache[chave] = medir_foto(self.imagem(arq), rois, ids,
                                                br["rgb"] if br else None, br["xy"] if br else None)
            return self._cache[chave]

        def tabela_leituras(self, so_confirmadas=True):
            linhas, anot = [], {}
            for arq in self.arquivos:
                if so_confirmadas and arq not in self.conf:
                    continue
                res, an = self.medir(arq)
                br = self.brancos.get(arq)
                for r in res:
                    linhas.append({"Arquivo": os.path.basename(arq),
                                   "Branco_ref": "nenhum" if not br else
                                   ("automatico" if br.get("auto") else "manual"), **r})
                if an is not None:
                    anot[arq] = an
            return pd.DataFrame(linhas), anot

        def respostas(self):
            """Resposta de cada ID: manual > foto confirmada (descontado o branco de reagente, se pedido)."""
            leit, _ = self.tabela_leituras()
            col = self.resp.get()
            out = {"foto": {}, "usada": {}, "origem": {}, "detalhe": {}, "branco_medio": float("nan"),
                   "n_brancos": 0, "desconto": False, "fora": [], "repetidos": []}
            if len(leit):
                eh_b = leit["ID"].map(self.eh_branco)
                br = leit[eh_b][col]
                out["n_brancos"] = len(br)
                out["branco_medio"] = float(br.mean()) if len(br) else float("nan")
                out["desconto"] = bool(self.usar_br.get() and self.descontar.get() and len(br))
                desc = out["branco_medio"] if out["desconto"] else 0.0
                am = leit[~eh_b]
                g = am.groupby("ID")
                out["foto"] = (g[col].mean() - desc).to_dict()
                out["repetidos"] = sorted(g.size()[g.size() > 1].index)
                for i, sub in g:
                    out["detalhe"][i] = {"Arquivo": ", ".join(sorted(set(sub["Arquivo"]))),
                                         "R": sub["R"].mean(), "G": sub["G"].mean(), "B": sub["B"].mean(),
                                         "R_branco": sub["R_branco"].mean(), "Bruto": sub[col].mean(),
                                         "Branco_ref": ", ".join(sorted(set(sub["Branco_ref"])))}
                plano = set(self.ids_plano())
                out["fora"] = sorted(i for i in out["foto"] if plano and i not in plano)
            alvos = self.ids_plano() or ordenar_ids(out["foto"])
            for e in alvos:
                m = num(self.manual.get(e, ""))
                if not math.isnan(m):
                    out["usada"][e], out["origem"][e] = m, "manual"
                elif e in out["foto"]:
                    out["usada"][e], out["origem"][e] = out["foto"][e], "foto"
            return out

        def calibracao(self, resp=None):
            resp = resp or self.respostas()
            xs, ys, ids = [], [], []
            for e, v in self.conc.items():
                c = num(v)
                y = resp["usada"].get(e)
                if not math.isnan(c) and y is not None and not math.isnan(y):
                    xs.append(c)
                    ys.append(y)
                    ids.append(e)
            cal = calibrar(xs, ys)
            cal["ids"] = ids
            return cal

        # ------------------------------------------------------------ desenho
        def atualizar_fotos(self):
            self.tab_fotos.delete(*self.tab_fotos.get_children())
            for i, a in enumerate(self.arquivos):
                br = self.brancos.get(a)
                self.tab_fotos.insert("", "end", iid=str(i), values=[
                    os.path.basename(a), "-".join(self.partes_validas(a)) or "(nenhum)",
                    len(self.rois.get(a, [])),
                    "nao" if not br else ("auto" if br.get("auto") else "manual"), self.situacao(a)])
            if self.atual in self.arquivos:
                self.idx = self.arquivos.index(self.atual)
                self.tab_fotos.selection_set(str(self.idx))
                self.tab_fotos.see(str(self.idx))
            self.mostrar_foto()
            self.atualizar_leituras()

        def mostrar_foto(self):
            if not self.atual:
                self.canvas.delete("all")
                self._chave_foto = None
                return
            img = self.imagem(self.atual)
            h, w = img.shape[:2]
            cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
            chave = (self.atual, cw, ch)
            if chave != self._chave_foto:
                self.fator = min(cw / w, ch / h)
                red = cv2.resize(img, (max(1, int(w * self.fator)), max(1, int(h * self.fator))),
                                 interpolation=cv2.INTER_AREA)
                self.foto = ImageTk.PhotoImage(Image.fromarray(red[..., ::-1]))
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.foto, tags="img")
                self._chave_foto = chave
            self.desenhar_rois()

        def desenhar_rois(self):
            self.canvas.delete("roi")
            if not self.atual:
                return
            f = self.fator
            rois, ids = self.ids_da_foto(self.atual)
            for p, i in zip(rois, ids):
                sel = p is self.sel
                cor = "#00e5ff" if self.eh_branco(i) else ("#ff5a36" if i.endswith("?") else "#ffe100")
                pontos = [c * f for v in p["pts"] for c in v]
                self.canvas.create_polygon(*pontos, outline=cor, fill="", width=3 if sel else 2, tags="roi")
                if sel:
                    for vx, vy in p["pts"]:
                        x, y = vx * f, vy * f
                        self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, outline="#ffffff",
                                                     fill=cor, width=1, tags="roi")
                xs = [v[0] * f for v in p["pts"]]
                ys = [v[1] * f for v in p["pts"]]
                tx, ty = sum(xs) / len(xs), min(ys) - 10
                self.canvas.create_text(tx + 1, ty + 1, text=i, fill="#000000",
                                        font=("TkDefaultFont", 11, "bold"), tags="roi")
                self.canvas.create_text(tx, ty, text=i, fill=cor,
                                        font=("TkDefaultFont", 11, "bold"), tags="roi")
            a = self._arrasto
            if a and a.get("tipo") == "novo" and a.get("moveu"):
                self.canvas.create_rectangle(a["x0"] * f, a["y0"] * f, a["x1"] * f, a["y1"] * f,
                                             outline="#ffe100", dash=(4, 3), width=2, tags="roi")
            br = self.brancos.get(self.atual)
            if br:
                x, y = br["xy"][0] * f, br["xy"][1] * f
                self.canvas.create_rectangle(x - 7, y - 7, x + 7, y + 7, outline="#22c55e", width=2, tags="roi")
                self.canvas.create_text(x + 12, y, anchor="w", fill="#22c55e", tags="roi",
                                        font=("TkDefaultFont", 10, "bold"),
                                        text="branco" + (" (auto)" if br.get("auto") else ""))

        def atualizar_leituras(self):
            self.tab_leit.delete(*self.tab_leit.get_children())
            if not self.atual:
                self.avisar("Abra as fotos dos tubos na aba 2.")
                return
            res, _ = self.medir(self.atual)
            resp = self.respostas()
            col = self.resp.get()
            self._linhas_rois = self.ids_da_foto(self.atual)[0]
            desc = resp["branco_medio"] if resp["desconto"] else 0.0
            for p, r in zip(self._linhas_rois, res):
                b = self.eh_branco(r["ID"])
                final = "" if b or r["ID"].endswith("?") else fmt(r[col] - desc)
                self.tab_leit.insert("", "end", values=[
                    r["ID"], self.conc.get(r["ID"], ""), f"{r['R']:.1f}", f"{r['G']:.1f}", f"{r['B']:.1f}",
                    fmt(r[col]), final])
            nconf = sum(1 for a in self.arquivos if a in self.conf)
            self.avisar(f"[{self.idx + 1}/{len(self.arquivos)}] {os.path.basename(self.atual)}: "
                        f"{self.situacao(self.atual)}   |   {nconf} de {len(self.arquivos)} foto(s) confirmada(s)"
                        + (f"   |   brancos de reagente: {resp['n_brancos']} "
                           f"(media {resp['branco_medio']:.3f})" if resp["n_brancos"] else ""))

        # ============================================================ ABA 3
        def _montar_aba3(self):
            b = Fluxo(self.aba3, padding=(8, 6, 8, 2))
            b.pack(fill="x")
            g = b.grupo("Resultados")
            ttk.Button(g, text="Gerar / atualizar", command=self.gerar_sinal).pack(side="left")
            g = b.grupo("Copiar")
            ttk.Button(g, text="Coluna da resposta", command=self.copiar_coluna).pack(side="left")
            ttk.Button(g, text="Tabela inteira", command=lambda: self.copiar_coluna(tudo=True)).pack(side="left", padx=4)
            g = b.grupo("Formato")
            ttk.Label(g, text="decimal:").pack(side="left")
            ttk.Combobox(g, textvariable=self.sep, values=["virgula", "ponto"], state="readonly",
                         width=8).pack(side="left", padx=(2, 8))
            ttk.Label(g, text="casas:").pack(side="left")
            ttk.Spinbox(g, from_=2, to=6, textvariable=self.casas, width=3).pack(side="left", padx=2)
            g = b.grupo("Exportar")
            ttk.Button(g, text="Salvar XLSX...", command=self.exportar_xlsx).pack(side="left")
            b.refluir(1200)

            corpo = ttk.Panedwindow(self.aba3, orient="horizontal")
            corpo.pack(fill="both", expand=True)
            esq = ttk.Frame(corpo)
            corpo.add(esq, weight=3)
            self.txt_sinal = tk.Text(esq, wrap="word", height=9, font=("TkDefaultFont", 9))
            self.txt_sinal.pack(fill="x", padx=6, pady=(6, 2))
            quadro = ttk.Frame(esq, padding=6)
            quadro.pack(fill="both", expand=True)
            self.tab_sinal = ttk.Treeview(quadro, columns=["Linha"], show="headings")
            sy = ttk.Scrollbar(quadro, orient="vertical", command=self.tab_sinal.yview)
            sx = ttk.Scrollbar(quadro, orient="horizontal", command=self.tab_sinal.xview)
            self.tab_sinal.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
            sy.pack(side="right", fill="y")
            sx.pack(side="bottom", fill="x")
            self.tab_sinal.pack(fill="both", expand=True)
            dir_ = ttk.Frame(corpo)
            corpo.add(dir_, weight=2)
            self.fig = Figure(figsize=(5.4, 4.4), dpi=100)
            self.fig_canvas = FigureCanvasTkAgg(self.fig, master=dir_)
            self.fig_canvas.get_tk_widget().pack(fill="both", expand=True)

        def linhas_resultado(self):
            """Tabela final: uma linha por ensaio (ou por ID lido, se nao houver planejamento)."""
            resp = self.respostas()
            cal = self.calibracao(resp)
            self.cal = cal
            col = self.resp.get()
            fatores = self.meta["nomes"] if self.meta else []
            if self.df is not None:
                base = [(int(r["Linha"]), r["Ensaio"], {n: r[n] for n in fatores})
                        for _, r in self.df.iterrows()]
            else:
                base = [(i + 1, e, {}) for i, e in enumerate(ordenar_ids(resp["usada"]))]
            linhas = []
            for linha, e, fat in base:
                d = resp["detalhe"].get(e, {}) if resp["origem"].get(e) == "foto" else {}
                y = resp["usada"].get(e, np.nan)
                c = num(self.conc.get(e, ""))
                linhas.append({"Linha": linha, "ID": e, **fat,
                               "Concentracao": c,
                               "Foto": d.get("Arquivo", ""),
                               "R": d.get("R", np.nan), "G": d.get("G", np.nan), "B": d.get("B", np.nan),
                               "R_branco": d.get("R_branco", np.nan),
                               col + " bruto": d.get("Bruto", np.nan),
                               "Resposta": y,
                               "Conc. prevista": (np.nan if not math.isnan(c) else prever_conc(cal, y)),
                               "Origem": resp["origem"].get(e, "sem leitura")})
            return pd.DataFrame(linhas), resp, cal

        def gerar_sinal(self, mostrar_erros=True):
            if self.df is None and not self.conf:
                if mostrar_erros:
                    messagebox.showinfo(APP, "Abra o planejamento na aba 1 ou confirme ao menos uma foto na aba 2.")
                return None
            t, resp, cal = self.linhas_resultado()
            self.sinal = t
            col = self.resp.get()
            cols = list(t.columns)
            self.tab_sinal.configure(columns=cols)
            for c in cols:
                self.tab_sinal.heading(c, text=c)
                self.tab_sinal.column(c, width=150 if c == "Foto" else 92, anchor="center")
            self.tab_sinal.delete(*self.tab_sinal.get_children())
            fatores = self.meta["nomes"] if self.meta else []
            for _, r in t.iterrows():
                vals = []
                for c in cols:
                    v = r[c]
                    if isinstance(v, str):
                        vals.append(v)
                    elif c == "Linha":
                        vals.append(int(v))
                    elif c in fatores:
                        vals.append("" if (isinstance(v, float) and math.isnan(v)) else f"{v:g}")
                    else:
                        vals.append(fmt(float(v), 1 if c in ("R", "G", "B", "R_branco") else 4))
                self.tab_sinal.insert("", "end", values=vals)
            n_foto = int((t["Origem"] == "foto").sum())
            n_man = int((t["Origem"] == "manual").sum())
            n_sem = int((t["Origem"] == "sem leitura").sum())
            L = [f"Resposta = {col}"
                 + (f" - media dos {resp['n_brancos']} brancos de reagente ({resp['branco_medio']:.4f})"
                    if resp["desconto"] else "")
                 + ".   A = -log10(sinal do tubo / branco de referencia).",
                 f"{n_foto} lido(s) em foto, {n_man} manual(is), {n_sem} sem leitura "
                 "(saem como '*', o simbolo de dado ausente do Minitab)."]
            L.append(texto_calibracao(cal, col))
            if cal["n"] >= 3:
                ordem = np.argsort(cal["x"])
                y = np.asarray(cal["y"])[ordem]
                if not (np.all(np.diff(y) >= 0) or np.all(np.diff(y) <= 0)):
                    L.append("AVISO: a resposta nao cresce (nem cai) de forma continua com a concentracao. "
                             "Reveja as ROIs e o branco dos pontos fora da tendencia.")
            pend = [os.path.basename(a) for a in self.arquivos if a not in self.conf]
            if pend:
                L.append("AVISO: fotos ainda nao confirmadas (fora do calculo): " + ", ".join(pend) + ".")
            auto = sorted({os.path.basename(a) for a in self.conf if self.brancos.get(a, {}).get("auto")})
            if auto:
                L.append("AVISO: branco de referencia automatico em: " + ", ".join(auto)
                         + ". Confira se o quadrado verde esta numa area clara e bem iluminada.")
            if resp["repetidos"]:
                L.append("AVISO: IDs lidos em mais de uma ROI (usada a media): " + ", ".join(resp["repetidos"]) + ".")
            if resp["fora"]:
                L.append("AVISO: IDs lidos que nao existem no planejamento: " + ", ".join(resp["fora"]) + ".")
            self.txt_sinal.delete("1.0", "end")
            self.txt_sinal.insert("1.0", "\n".join(L))
            desc = [(prever_conc(cal, r["Resposta"]), r["Resposta"]) for _, r in t.iterrows()
                    if math.isnan(num(self.conc.get(r["ID"], ""))) and pd.notna(r["Resposta"])]
            desc = [(x, y) for x, y in desc if not math.isnan(x)]
            try:
                grafico_calibracao(self.fig, cal, desc, col)
            except Exception as e:
                self.fig.clear()
                self.fig.text(0.5, 0.5, f"Nao foi possivel desenhar:\n{e}", ha="center", va="center")
            self.fig_canvas.draw()
            return t

        def numero(self, v, casas=None):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "*"
            s = f"{v:.{int(self.casas.get()) if casas is None else casas}f}"
            return s.replace(".", ",") if self.sep.get() == "virgula" else s

        def copiar_coluna(self, tudo=False):
            t = self.gerar_sinal()
            if t is None:
                return
            if tudo:
                linhas = ["\t".join(t.columns)]
                for _, r in t.iterrows():
                    linhas.append("\t".join(v if isinstance(v, str) else
                                            (str(int(v)) if c == "Linha" else self.numero(float(v)))
                                            for c, v in r.items()))
                texto = "\n".join(linhas)
            else:
                texto = "\n".join(self.numero(v) for v in t["Resposta"])
            self.clipboard_clear()
            self.clipboard_append(texto)
            self.update()
            n = int(t["Resposta"].notna().sum())
            self.avisar("Tabela inteira copiada." if tudo else
                        f"Coluna copiada: {len(t)} linhas ({n} com valor).")

        def exportar_xlsx(self):
            t = self.gerar_sinal()
            if t is None:
                return
            arq = filedialog.asksaveasfilename(title="Salvar planilha", defaultextension=".xlsx",
                                               initialfile="resultados.xlsx",
                                               filetypes=[("Planilha do Excel", "*.xlsx")])
            if not arq:
                return
            leit, _ = self.tabela_leituras()
            cal = self.cal
            col = self.resp.get()
            cal_df = pd.DataFrame({"Concentracao": list(cal["x"]), col: list(cal["y"]),
                                   "ID": cal.get("ids", [])})
            resumo = pd.DataFrame({
                "Parametro": ["Resposta", "n pontos", "Intercepto (a)", "Inclinacao (b)", "R2",
                              "S(y/x)", "EP(b)", "LOD", "LOQ"],
                "Valor": [col, cal["n"], cal["a"], cal["b"], cal["R2"], cal["Syx"],
                          cal["EP_b"], cal["LOD"], cal["LOQ"]]})
            notas = self.txt_sinal.get("1.0", "end").strip().split("\n")
            try:
                with pd.ExcelWriter(arq, engine="openpyxl") as xl:
                    t.to_excel(xl, sheet_name="Resultados", index=False)
                    if len(leit):
                        leit.to_excel(xl, sheet_name="Leituras RGB", index=False)
                    cal_df.to_excel(xl, sheet_name="Calibracao", index=False, startrow=0)
                    resumo.to_excel(xl, sheet_name="Calibracao", index=False, startcol=4)
                    pd.DataFrame({"Notas": notas}).to_excel(xl, sheet_name="Notas", index=False)
                    if len(cal_df) >= 2:
                        self._grafico_xlsx(xl.sheets["Calibracao"], len(cal_df), col)
            except OSError as e:
                messagebox.showerror(APP, f"Nao foi possivel salvar (arquivo aberto no Excel?):\n{e}")
                return
            self.avisar(f"Planilha salva em {arq}")
            messagebox.showinfo(APP, f"Planilha salva em:\n{arq}")

        def _grafico_xlsx(self, ws, n, col):
            from openpyxl.chart import Reference, ScatterChart, Series
            from openpyxl.chart.trendline import Trendline
            ch = ScatterChart()
            ch.title = "Curva de calibracao"
            ch.style = 13
            ch.x_axis.title = "Concentracao"
            ch.y_axis.title = col
            ch.height, ch.width = 9, 15
            xref = Reference(ws, min_col=1, min_row=2, max_row=n + 1)
            yref = Reference(ws, min_col=2, min_row=1, max_row=n + 1)
            s = Series(yref, xref, title_from_data=True)
            s.marker.symbol = "circle"
            s.graphicalProperties.line.noFill = True
            s.trendline = Trendline(trendlineType="linear", dispRSqr=True, dispEq=True)
            ch.series.append(s)
            ws.add_chart(ch, "A" + str(n + 4))

        # ----------------------------------------------------------- sessao
        def salvar_sessao(self):
            arq = filedialog.asksaveasfilename(title="Salvar sessao", defaultextension=".json",
                                               filetypes=[("Sessao", "*.json")])
            if not arq:
                return
            dados = {"fonte": self.fonte, "caminho_fonte": self.caminho_fonte,
                     "planilha": self.planilha.get(), "id_col": self.id_col.get(),
                     "manual": self.manual, "conc": self.conc, "col_conc": self.col_conc.get(),
                     "arquivos": self.arquivos, "rois": self.rois, "brancos": self.brancos,
                     "confirmadas": sorted(self.conf), "lado": self.lado.get(),
                     "resposta": self.resp.get(), "usar_br": bool(self.usar_br.get()),
                     "descontar": bool(self.descontar.get()), "auto_num": bool(self.auto_num.get()),
                     "n_inicial": self.n_inicial.get(), "sep": self.sep.get(), "casas": self.casas.get()}
            try:
                with open(arq, "w", encoding="utf-8") as f:
                    json.dump(dados, f, indent=1, ensure_ascii=False)
            except (OSError, TypeError, ValueError) as e:
                messagebox.showerror(APP, f"Nao foi possivel salvar:\n{e}")
                return
            self.avisar(f"Sessao salva em {arq}")

        def abrir_sessao(self):
            arq = filedialog.askopenfilename(title="Abrir sessao", filetypes=[("Sessao", "*.json")])
            if not arq:
                return
            try:
                with open(arq, encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError) as e:
                messagebox.showerror(APP, f"Nao foi possivel abrir a sessao:\n{e}")
                return
            self.lado.set(d.get("lado", LADO_PADRAO))
            self.resp.set(d.get("resposta", "A_R"))
            self.usar_br.set(d.get("usar_br", False))
            self.descontar.set(d.get("descontar", True))
            self.auto_num.set(d.get("auto_num", True))
            self.n_inicial.set(d.get("n_inicial", 1))
            self.casas.set(d.get("casas", 4))
            faltam = [a for a in d.get("arquivos", []) if not os.path.exists(a)]
            self.arquivos = [a for a in d.get("arquivos", []) if a not in faltam]
            self.rois = {}
            for a in self.arquivos:
                if a in d.get("rois", {}):
                    self.rois[a] = d["rois"][a]
                else:                       # sessao antiga: ROIs circulares {x, y, r}
                    esc = 1.0
                    try:
                        esc = self.imagem(a).shape[1] / LARGURA_REF
                    except IOError:
                        pass
                    self.rois[a] = [{"id": c.get("id"),
                                     "pts": retangulo(c["x"], c["y"],
                                                      2 * (c.get("r") or LADO_PADRAO / 2) * esc)}
                                    for c in d.get("pontos", {}).get(a, [])]
            self.brancos = {a: b for a, b in d.get("brancos", {}).items() if a in self.arquivos}
            self.conf = {a for a in d.get("confirmadas", []) if a in self.arquivos}
            self.imagens, self._cache, self.hist = {}, {}, {}
            self.idx, self.atual, self.sel = 0, (self.arquivos[0] if self.arquivos else None), None
            self.df = None
            if d.get("fonte"):
                self.fonte, self.caminho_fonte = d["fonte"], d.get("caminho_fonte", "")
                self.cb_plan.configure(values=[p["nome"] for p in self.fonte["planilhas"]])
                self.planilha.set(d.get("planilha", ""))
                self.id_col.set(d.get("id_col", ""))
                self.col_conc.set(d.get("col_conc", SEM_CONC))
                self.montar(manter_id=True, manual=d.get("manual"), conc=d.get("conc"))
            else:
                self.manual = d.get("manual", {})
                self.conc = d.get("conc", {})
                self.atualizar_fotos()
            self.sep.set(d.get("sep", "virgula"))
            self.alternar_branco_reagente()
            if faltam:
                messagebox.showwarning(APP, "Fotos nao encontradas (foram ignoradas):\n" + "\n".join(faltam))

    App().mainloop()


if __name__ == "__main__":
    rodar_gui()
