"""Interface Gradio pour Laplace Immo.

Simulateur de prix de maisons consommant l'API FastAPI (api/main.py).
Direction design : hybride Compass (ivoire/serif) + Stripe (clean tech).

Lancement : python ui/app.py
Acces     : http://localhost:7860
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import requests

chemin_projet = Path(__file__).resolve().parent.parent
if str(chemin_projet) not in sys.path:
    sys.path.insert(0, str(chemin_projet))

from ui.defaults import (  # noqa: E402
    DEFAULTS,
    NEIGHBORHOODS,
    ORDRE_QUALITES,
    QUALITES_LABELS,
)

# ==================== Configuration ====================

URL_API = "http://localhost:8000"
TIMEOUT_API = 10


# ==================== Communication API ====================


def estimer_prix(
    neighborhood,
    grlivarea,
    overallqual,
    yearbuilt,
    totalbsmtsf,
    firstflrsf,
    garagearea,
    fullbath,
    bedrooms,
    exterqual,
    yearremod,
    overallcond,
    housestyle,
    bsmtqual,
    bsmtexposure,
    garagefinish,
    kitchenqual,
    openporchsf,
    has_pool,
    pool_area_input,
    has_extra_fp,
    extra_fp_count,
    has_big_garage,
    big_garage_cars,
    has_big_deck,
    big_deck_area,
):
    """Construit le payload, appelle POST /predict, retourne du HTML pour gr.HTML.

    Les champs principaux sont obligatoires : s'il en manque un, on affiche
    une erreur sans appeler l'API. Les champs detailles laisses vides sont
    remplaces par les valeurs typiques du dataset (DEFAULTS).
    """
    # ----- Champs principaux obligatoires -----
    obligatoires = {
        "Quartier": neighborhood,
        "Surface habitable": grlivarea,
        "Qualite generale": overallqual,
        "Annee de construction": yearbuilt,
        "Surface sous-sol": totalbsmtsf,
        "Surface RDC": firstflrsf,
        "Surface du garage": garagearea,
        "Salles de bain completes": fullbath,
        "Chambres": bedrooms,
        "Qualite exterieure": exterqual,
    }
    manquants = [nom for nom, valeur in obligatoires.items() if valeur is None]
    if manquants:
        return rendre_erreur(
            "Champs a renseigner avant l'estimation : " + ", ".join(manquants) + "."
        )

    # ----- Tolerance de saisie : un 0 dans un champ optionnel dont l'echelle
    # ne contient pas 0 (note de 1 a 10, annee) vaut "non renseigne" -----
    if overallcond == 0:
        overallcond = None
    if yearremod == 0:
        yearremod = None

    # ----- Verification des bornes (les optionnels vides sont ignores) -----
    bornes = [
        ("Surface habitable", grlivarea, 100, 10000),
        ("Qualite generale", overallqual, 1, 10),
        ("Annee de construction", yearbuilt, 1800, 2030),
        ("Surface sous-sol", totalbsmtsf, 0, 10000),
        ("Surface RDC", firstflrsf, 100, 8000),
        ("Surface du garage", garagearea, 0, 3000),
        ("Salles de bain completes", fullbath, 0, 6),
        ("Chambres", bedrooms, 0, 12),
        ("Annee derniere renovation", yearremod, 1800, 2030),
        ("Etat general", overallcond, 1, 10),
        ("Porche ouvert", openporchsf, 0, 1000),
    ]
    if has_pool:
        bornes.append(("Surface piscine", pool_area_input, 1, 2000))
    if has_extra_fp:
        bornes.append(("Nombre de cheminees", extra_fp_count, 1, 4))
    if has_big_garage:
        bornes.append(("Places de garage", big_garage_cars, 1, 5))
    if has_big_deck:
        bornes.append(("Surface terrasse", big_deck_area, 1, 1500))

    hors_limites = [
        f"{nom} : entre {mini} et {maxi}"
        for nom, valeur, mini, maxi in bornes
        if valeur is not None and not (mini <= valeur <= maxi)
    ]
    if hors_limites:
        return rendre_erreur("Valeurs hors limites. " + " ; ".join(hors_limites) + ".")

    payload = dict(DEFAULTS)
    payload.update(
        {
            "Neighborhood": neighborhood,
            "GrLivArea": int(grlivarea),
            "OverallQual": int(overallqual),
            "YearBuilt": int(yearbuilt),
            "TotalBsmtSF": int(totalbsmtsf),
            "1stFlrSF": int(firstflrsf),
            "GarageArea": int(garagearea),
            "FullBath": int(fullbath),
            "BedroomAbvGr": int(bedrooms),
            "ExterQual": exterqual,
        }
    )

    # ----- Champs detailles : pris en compte seulement s'ils sont renseignes -----
    # Une maison jamais renovee a une annee de renovation egale a sa construction.
    if yearremod is None:
        payload["YearRemodAdd"] = int(yearbuilt)
    else:
        # Une renovation ne peut pas etre anterieure a la construction
        payload["YearRemodAdd"] = int(max(yearremod, yearbuilt))
    if overallcond is not None:
        payload["OverallCond"] = int(overallcond)
    if housestyle is not None:
        payload["HouseStyle"] = housestyle
    if bsmtqual is not None:
        payload["BsmtQual"] = bsmtqual
    if bsmtexposure is not None:
        payload["BsmtExposure"] = bsmtexposure
    if garagefinish is not None:
        payload["GarageFinish"] = garagefinish
    if kitchenqual is not None:
        payload["KitchenQual"] = kitchenqual
    if openporchsf is not None:
        payload["OpenPorchSF"] = int(openporchsf)

    if has_pool:
        payload["PoolArea"] = int(pool_area_input)
        payload["PoolQC"] = "Gd"
    if has_extra_fp:
        payload["Fireplaces"] = int(extra_fp_count)
        payload["FireplaceQu"] = "Gd"
    if has_big_garage:
        payload["GarageCars"] = int(big_garage_cars)
    if has_big_deck:
        payload["WoodDeckSF"] = int(big_deck_area)

    try:
        r = requests.post(f"{URL_API}/predict", json=payload, timeout=TIMEOUT_API)
        if r.status_code == 200:
            return rendre_resultat(r.json()["prix_estime_dollars"])
        if r.status_code == 422:
            return rendre_erreur("Donnees invalides. Verifie les valeurs saisies.")
        return rendre_erreur(f"Erreur API (HTTP {r.status_code}).")
    except requests.ConnectionError:
        return rendre_erreur(
            "API inaccessible. Lance-la avec : uvicorn api.main:app --reload"
        )
    except requests.Timeout:
        return rendre_erreur("L'API ne repond pas (timeout).")
    except Exception as e:  # pragma: no cover
        return rendre_erreur(f"Erreur inattendue : {e}")


# ==================== Templates HTML ====================


def rendre_header() -> str:
    return """
    <div class="lp-header">
        <div class="lp-brand">
            <div class="lp-brand-mark">Laplace</div>
            <div class="lp-brand-name">Immo</div>
        </div>
        <div class="lp-brand-tag">Estimation immobiliere par intelligence artificielle</div>
    </div>
    """


def rendre_placeholder() -> str:
    return """
    <div class="lp-result lp-result-empty">
        <div class="lp-eyebrow">Estimation Laplace</div>
        <div class="lp-price lp-price-empty">$ —</div>
        <div class="lp-empty-text">
            Renseignez les caracteristiques de la maison ci-dessus, puis lancez
            l'estimation pour obtenir un prix.
        </div>
    </div>
    """


def rendre_resultat(prix: float) -> str:
    prix_min = prix * 0.93
    prix_max = prix * 1.07
    return f"""
    <div class="lp-result">
        <div class="lp-eyebrow">Estimation Laplace</div>
        <div class="lp-price">$ {prix:,.0f}</div>
        <div class="lp-confidence-bar">
            <div class="lp-confidence-fill"></div>
        </div>
        <div class="lp-range">
            <div class="lp-range-row">
                <span class="lp-range-label">Fourchette basse</span>
                <span class="lp-range-value">$ {prix_min:,.0f}</span>
            </div>
            <div class="lp-range-row">
                <span class="lp-range-label">Fourchette haute</span>
                <span class="lp-range-value">$ {prix_max:,.0f}</span>
            </div>
        </div>
        <div class="lp-factors-title">Facteurs determinants</div>
        <ul class="lp-factors">
            <li>Surface habitable totale</li>
            <li>Qualite generale de la maison</li>
            <li>Quartier et localisation</li>
        </ul>
    </div>
    """


def rendre_erreur(msg: str) -> str:
    return f'<div class="lp-error">{msg}</div>'


# ==================== CSS custom ====================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    max-width: 920px !important;
    background: #FAFAF7 !important;
    padding: 1rem 2rem 4rem 2rem !important;
}

.gradio-container * {
    font-family: 'Inter', system-ui, sans-serif;
}

.lp-header {
    padding: 1.5rem 0 2.5rem 0;
    border-bottom: 1px solid #E7E4DC;
    margin-bottom: 2rem;
    text-align: center;
}

.lp-brand {
    display: flex;
    align-items: baseline;
    justify-content: center;
    gap: 0.5rem;
}

.lp-brand-mark {
    font-family: 'Fraunces', serif;
    font-weight: 400;
    font-size: 2.4rem;
    color: #0E1116;
    letter-spacing: -0.01em;
    line-height: 1;
}

.lp-brand-name {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #1F3A5F;
    font-style: italic;
    letter-spacing: -0.01em;
    line-height: 1;
}

.lp-brand-tag {
    font-family: 'Inter', sans-serif;
    color: #5B6770;
    font-size: 0.9rem;
    font-weight: 400;
    margin-top: 0.8rem;
}

.lp-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #C9A96E;
    font-weight: 700;
    margin-bottom: 0.8rem;
    text-align: center;
}

.lp-result {
    background: #FFFFFF;
    border: 1px solid #E7E4DC;
    border-radius: 16px;
    padding: 2.5rem 2.2rem;
    margin-top: 2rem;
    box-shadow:
        0 1px 2px rgba(0, 0, 0, 0.03),
        0 8px 24px rgba(15, 20, 25, 0.04);
    animation: lpFadeIn 400ms ease-out;
    text-align: center;
}

.lp-result-empty {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFAF7 100%);
    border: 1px dashed #D6D3CC;
    box-shadow: none;
}

@keyframes lpFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.lp-price {
    font-family: 'Fraunces', serif !important;
    font-weight: 400;
    font-size: 4.5rem;
    line-height: 1;
    color: #0E1116;
    margin: 0 0 1.5rem 0;
    letter-spacing: -0.03em;
    text-align: center;
}

.lp-price-empty {
    color: #C7C2B6;
}

.lp-empty-text {
    color: #5B6770;
    font-size: 0.95rem;
    line-height: 1.65;
    max-width: 480px;
    margin: 0 auto;
}

.lp-confidence-bar {
    height: 3px;
    background: #F1EDE3;
    border-radius: 2px;
    margin: 0 auto 1.5rem auto;
    overflow: hidden;
    max-width: 320px;
}

.lp-confidence-fill {
    height: 100%;
    width: 78%;
    background: linear-gradient(90deg, #C9A96E 0%, #B8965E 100%);
    border-radius: 2px;
}

.lp-range {
    background: #FAFAF7;
    border: 1px solid #EFECE4;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0 auto 1.8rem auto;
    max-width: 420px;
    text-align: left;
}

.lp-range-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.4rem 0;
}

.lp-range-row + .lp-range-row {
    border-top: 1px solid #EFECE4;
}

.lp-range-label {
    font-size: 0.82rem;
    color: #5B6770;
    font-weight: 500;
}

.lp-range-value {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #0E1116;
}

.lp-factors-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #5B6770;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    text-align: center;
}

.lp-factors {
    list-style: none;
    padding: 0;
    margin: 0 auto;
    max-width: 360px;
    text-align: left;
}

.lp-factors li {
    padding: 0.6rem 0;
    border-bottom: 1px solid #EFECE4;
    font-size: 0.95rem;
    color: #0E1116;
    font-weight: 500;
    display: flex;
    align-items: center;
}

.lp-factors li:last-child {
    border-bottom: none;
}

.lp-factors li::before {
    content: "";
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #C9A96E;
    margin-right: 0.8rem;
    flex-shrink: 0;
}

.lp-error {
    color: #991B1B;
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    font-size: 0.95rem;
    margin-top: 2rem;
}

button.lg.primary {
    background: #1F3A5F !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.95rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 200ms ease !important;
    box-shadow: 0 1px 2px rgba(31, 58, 95, 0.15) !important;
    margin-top: 1rem !important;
}

button.lg.primary:hover {
    background: #142844 !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(31, 58, 95, 0.22) !important;
}

footer {display: none !important;}
.show-api {display: none !important;}
.built-with-gradio {display: none !important;}
.svelte-1ipelgc {display: none !important;}
"""


# ==================== Theme Gradio ====================

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#FAFAF7",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#FAFAF7",
    border_color_primary="#E7E4DC",
    block_border_color="#E7E4DC",
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 1px 2px rgba(0,0,0,0.03)",
    block_label_text_weight="600",
    block_label_text_color="#0E1116",
    block_label_text_size="*text_sm",
    input_border_color="#E7E4DC",
    input_border_color_focus="#1F3A5F",
    input_radius="10px",
    button_primary_background_fill="#1F3A5F",
    button_primary_background_fill_hover="#142844",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#1F3A5F",
    button_primary_border_color_hover="#142844",
    slider_color="#1F3A5F",
)


# ==================== Construction de l'interface ====================


def _toggle_visibility(checked):
    return gr.update(visible=bool(checked))


def construire_interface() -> gr.Blocks:
    with gr.Blocks(
        theme=THEME,
        css=CUSTOM_CSS,
        title="Laplace Immo - Simulateur de prix",
    ) as demo:

        gr.HTML(rendre_header())

        gr.Markdown("### Caracteristiques principales")

        with gr.Row():
            with gr.Column():
                neighborhood = gr.Dropdown(
                    label="Quartier",
                    choices=NEIGHBORHOODS,
                    value=None,
                )
                grlivarea = gr.Number(
                    label="Surface habitable (sq ft)",
                    value=None,
                    precision=0,
                )
                overallqual = gr.Number(
                    label="Qualite generale (1-10)",
                    value=None,
                    precision=0,
                )
                yearbuilt = gr.Number(
                    label="Annee de construction",
                    value=None,
                    precision=0,
                )
                totalbsmtsf = gr.Number(
                    label="Surface sous-sol (sq ft)",
                    value=None,
                    precision=0,
                )

            with gr.Column():
                firstflrsf = gr.Number(
                    label="Surface RDC (sq ft)",
                    value=None,
                    precision=0,
                )
                garagearea = gr.Number(
                    label="Surface du garage (sq ft)",
                    value=None,
                    precision=0,
                )
                fullbath = gr.Number(
                    label="Salles de bain completes",
                    value=None,
                    precision=0,
                )
                bedrooms = gr.Number(
                    label="Chambres",
                    value=None,
                    precision=0,
                )
                exterqual = gr.Dropdown(
                    label="Qualite exterieure",
                    choices=[
                        (f"{q} - {QUALITES_LABELS[q]}", q) for q in ORDRE_QUALITES
                    ],
                    value=None,
                )

        with gr.Accordion("Caracteristiques detaillees", open=False):
            gr.Markdown(
                "Champs optionnels : laissez vide ce que vous ne connaissez pas, "
                "des valeurs typiques d'Ames seront utilisees."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Construction**")
                    yearremod = gr.Number(
                        label="Annee derniere renovation",
                        value=None,
                        precision=0,
                    )
                    overallcond = gr.Number(
                        label="Etat general (1-10)",
                        value=None,
                        precision=0,
                    )
                    housestyle = gr.Dropdown(
                        label="Style de maison",
                        choices=[
                            "1Story",
                            "1.5Fin",
                            "1.5Unf",
                            "2Story",
                            "2.5Fin",
                            "2.5Unf",
                            "SFoyer",
                            "SLvl",
                        ],
                        value=None,
                    )
                    gr.Markdown("**Sous-sol**")
                    bsmtqual = gr.Dropdown(
                        label="Qualite du sous-sol",
                        choices=["None"] + ORDRE_QUALITES,
                        value=None,
                    )
                    bsmtexposure = gr.Dropdown(
                        label="Exposition du sous-sol",
                        choices=["None", "No", "Mn", "Av", "Gd"],
                        value=None,
                    )

                with gr.Column():
                    gr.Markdown("**Garage**")
                    garagefinish = gr.Dropdown(
                        label="Finition du garage",
                        choices=["None", "Unf", "RFn", "Fin"],
                        value=None,
                    )
                    gr.Markdown("**Cuisine**")
                    kitchenqual = gr.Dropdown(
                        label="Qualite cuisine",
                        choices=ORDRE_QUALITES,
                        value=None,
                    )
                    gr.Markdown("**Exterieur**")
                    openporchsf = gr.Number(
                        label="Porche ouvert (sq ft)",
                        value=None,
                        precision=0,
                    )

        with gr.Accordion("Caracteristiques particulieres", open=False):
            gr.Markdown(
                "Cochez les cases si la maison possede ces caracteristiques. "
                "Les champs detailles apparaitront."
            )

            with gr.Row():
                with gr.Column():
                    has_pool = gr.Checkbox(label="Piscine")
                    pool_area_input = gr.Number(
                        label="Surface piscine (sq ft)",
                        value=400,
                        precision=0,
                        visible=False,
                    )

                    has_extra_fp = gr.Checkbox(label="Plus de 2 cheminees")
                    extra_fp_count = gr.Number(
                        label="Nombre de cheminees",
                        value=2,
                        precision=0,
                        visible=False,
                    )

                with gr.Column():
                    has_big_garage = gr.Checkbox(label="Garage XL (3+ places)")
                    big_garage_cars = gr.Number(
                        label="Places de garage",
                        value=3,
                        precision=0,
                        visible=False,
                    )

                    has_big_deck = gr.Checkbox(label="Grande terrasse bois")
                    big_deck_area = gr.Number(
                        label="Surface terrasse (sq ft)",
                        value=300,
                        precision=0,
                        visible=False,
                    )

            has_pool.change(_toggle_visibility, inputs=has_pool, outputs=pool_area_input)
            has_extra_fp.change(_toggle_visibility, inputs=has_extra_fp, outputs=extra_fp_count)
            has_big_garage.change(
                _toggle_visibility, inputs=has_big_garage, outputs=big_garage_cars
            )
            has_big_deck.change(
                _toggle_visibility, inputs=has_big_deck, outputs=big_deck_area
            )

        submit = gr.Button("Estimer le prix", variant="primary", size="lg")

        resultat = gr.HTML(rendre_placeholder())

        submit.click(
            fn=estimer_prix,
            inputs=[
                neighborhood,
                grlivarea,
                overallqual,
                yearbuilt,
                totalbsmtsf,
                firstflrsf,
                garagearea,
                fullbath,
                bedrooms,
                exterqual,
                yearremod,
                overallcond,
                housestyle,
                bsmtqual,
                bsmtexposure,
                garagefinish,
                kitchenqual,
                openporchsf,
                has_pool,
                pool_area_input,
                has_extra_fp,
                extra_fp_count,
                has_big_garage,
                big_garage_cars,
                has_big_deck,
                big_deck_area,
            ],
            outputs=resultat,
        )

    return demo


# ==================== Lancement ====================

if __name__ == "__main__":
    demo = construire_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_api=False,
        share=False,
    )
