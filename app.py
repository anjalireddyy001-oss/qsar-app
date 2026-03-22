from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

PAPER_COMPOUNDS = [
    {"id": 1,  "name": "SA-1",  "smiles": "CNCc1c(-c2ccccc2)n(-c2ccc(Cl)cc2)c(C)c1",               "activity": 0.2405},
    {"id": 2,  "name": "SA-2",  "smiles": "CNCc1c(-c2ccccc2Cl)n(-c2ccc(Cl)cc2)c(C)c1",             "activity": 0.2325},
    {"id": 3,  "name": "SA-3",  "smiles": "CNCc1c(-c2ccccc2Cl)n(-c2ccc(C)cc2)c(C)c1",              "activity": 0.2505},
    {"id": 4,  "name": "SA-4",  "smiles": "CNCc1c(-c2ccc(Cl)cc2)n(-c2ccccc2)c(C)c1",               "activity": 0.2500},
    {"id": 5,  "name": "SA-5",  "smiles": "CNCc1c(-c2ccc(Cl)cc2)n(-c2ccc(Cl)cc2)c(C)c1",           "activity": 0.2230},
    {"id": 6,  "name": "SA-6",  "smiles": "CNCc1c(-c2ccc(Cl)cc2)n(-c2ccc(C)cc2)c(C)c1",            "activity": 0.2590},
    {"id": 7,  "name": "SA-7",  "smiles": "CNCc1c(-c2ccccc2Cl)n(-c2ccccc2)c(C)c1",                 "activity": 0.2770},
    {"id": 8,  "name": "SA-8",  "smiles": "CNCc1c(-c2ccccc2Cl)n(-c2ccc(Cl)cc2)c(C)c1",             "activity": 0.2625},
    {"id": 9,  "name": "SA-9",  "smiles": "CNCc1c(-c2ccccc2Cl)n(-c2ccc(C)cc2)c(C)c1",              "activity": 0.2820},
    {"id": 10, "name": "SA-10", "smiles": "CNCc1c(-c2ccccc2F)n(-c2ccccc2)c(C)c1",                  "activity": 0.3175},
    {"id": 11, "name": "SA-11", "smiles": "CNCc1c(-c2ccccc2F)n(-c2ccc(Cl)cc2)c(C)c1",              "activity": 0.2845},
    {"id": 12, "name": "SA-12", "smiles": "CNCc1c(-c2ccccc2F)n(-c2ccc(C)cc2)c(C)c1",               "activity": 0.3190},
    {"id": 13, "name": "BM212", "smiles": "CN1CCN(Cc2c(-c3ccc(Cl)cc3)n(-c3ccc(Cl)cc3)c(C)c2)CC1", "activity": 0.6715},
]

DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
    'TPSA', 'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
    'FractionCSP3', 'HeavyAtomCount'
]

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt':             round(Descriptors.MolWt(mol), 3),
        'MolLogP':           round(Descriptors.MolLogP(mol), 3),
        'NumHDonors':        rdMolDescriptors.CalcNumHBD(mol),
        'NumHAcceptors':     rdMolDescriptors.CalcNumHBA(mol),
        'TPSA':              round(Descriptors.TPSA(mol), 3),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'RingCount':         rdMolDescriptors.CalcNumRings(mol),
        'NumAromaticRings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'FractionCSP3':      round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        'HeavyAtomCount':    mol.GetNumHeavyAtoms()
    }

def run_qsar(compounds):
    valid = [c for c in compounds if c.get('activity') is not None and c.get('descriptors')]
    if len(valid) < 3:
        return {'error': 'Need at least 3 compounds with activity values'}
    X = np.array([[c['descriptors'][d] for d in DESCRIPTOR_NAMES] for c in valid])
    y = np.array([float(c['activity']) for c in valid])
    best_r2 = -np.inf
    best_combo = None
    best_model = None
    best_scaler = None
    for n in range(1, min(3, len(valid) - 1) + 1):
        for combo in combinations(range(X.shape[1]), n):
            Xi = X[:, combo]
            sc = StandardScaler()
            Xs = sc.fit_transform(Xi)
            m = LinearRegression().fit(Xs, y)
            r2 = r2_score(y, m.predict(Xs))
            if r2 > best_r2:
                best_r2 = r2
                best_combo = combo
                best_model = m
                best_scaler = sc
    Xi = X[:, best_combo]
    Xs = best_scaler.transform(Xi)
    yp = best_model.predict(Xs)
    y_cv = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(Xi):
        sc = StandardScaler()
        m = LinearRegression().fit(sc.fit_transform(Xi[tr]), y[tr])
        y_cv[te] = m.predict(sc.transform(Xi[te]))
    desc_used = [DESCRIPTOR_NAMES[i] for i in best_combo]
    terms = [f"{best_model.coef_[i]:.4f}x{desc_used[i]}" for i in range(len(desc_used))]
    equation = "Absorbance = " + " + ".join(terms) + f" + ({best_model.intercept_:.4f})"
    return {
        'equation': equation,
        'r2': round(float(best_r2), 4),
        'q2': round(float(r2_score(y, y_cv)), 4),
        'descriptors_used': desc_used,
        'predictions': [
            {
                'smiles': valid[i]['smiles'],
                'name': valid[i].get('name', f'Compound {i+1}'),
                'actual': round(float(y[i]), 4),
                'predicted': round(float(yp[i]), 4),
                'residual': round(float(y[i] - yp[i]), 4)
            }
            for i in range(len(valid))
        ]
    }

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/load_paper', methods=['GET'])
def load_paper():
    compounds = []
    for c in PAPER_COMPOUNDS:
        desc = calculate_descriptors(c['smiles'])
        compounds.append({
            'id': c['id'],
            'name': c['name'],
            'smiles': c['smiles'],
            'activity': c['activity'],
            'activity_unit': 'Absorbance (ELISA)',
            'descriptors': desc
        })
    return jsonify({'compounds': compounds})

@app.route('/api/descriptors', methods=['POST'])
def get_descriptors():
    data = request.json
    smiles = data.get('smiles', '')
    desc = calculate_descriptors(smiles)
    if desc is None:
        return jsonify({'error': 'Invalid SMILES'}), 400
    return jsonify(desc)

@app.route('/api/qsar', methods=['POST'])
def run_qsar_analysis():
    data = request.json
    compounds = data.get('compounds', [])
    result = run_qsar(compounds)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)