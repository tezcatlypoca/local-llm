"""
Routes pour l'ajout et l'upload de documents dans le RAG.
"""

import os
import json
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from src.routes.rag.config import get_rag_manager, allowed_file, UPLOAD_FOLDER

logger = logging.getLogger(__name__)

# Créer le blueprint pour les routes de documents
documents_bp = Blueprint('rag_documents', __name__)


@documents_bp.route('/documents', methods=['POST'])
def add_document():
    """
    Ajoute un document à la base de connaissances.
    
    Body JSON:
    {
        "file_path": "/chemin/vers/fichier.pdf"  // OU
        "text": "Contenu texte direct",
        "source": "manual_input",  // optionnel
        "metadata": {},  // optionnel
        "document_id": "custom_id"  // optionnel
    }
    
    Returns:
        JSON avec le statut et le nombre de chunks ajoutés
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Body JSON requis"
            }), 400
        
        # Cas 1: Ajout via file_path
        if 'file_path' in data:
            file_path = data['file_path']
            metadata = data.get('metadata')
            document_id = data.get('document_id')
            
            try:
                chunks_count = rag_manager.add_document(
                    file_path=file_path,
                    metadata=metadata,
                    document_id=document_id
                )
                return jsonify({
                    "status": "success",
                    "message": "Document ajouté avec succès",
                    "file_path": file_path,
                    "chunks_added": chunks_count
                }), 200
            except FileNotFoundError as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 404
            except ValueError as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 400
            except PermissionError as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 403
        
        # Cas 2: Ajout via texte direct
        elif 'text' in data:
            text = data['text']
            source = data.get('source', 'manual_input')
            metadata = data.get('metadata')
            document_id = data.get('document_id')
            
            if not isinstance(text, str) or not text.strip():
                return jsonify({
                    "status": "error",
                    "message": "Le champ 'text' doit être une chaîne non vide"
                }), 400
            
            chunks_count = rag_manager.add_text(
                text=text,
                source=source,
                metadata=metadata,
                document_id=document_id
            )
            return jsonify({
                "status": "success",
                "message": "Texte ajouté avec succès",
                "chunks_added": chunks_count
            }), 200
        
        else:
            return jsonify({
                "status": "error",
                "message": "Le body doit contenir soit 'file_path' soit 'text'"
            }), 400
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout du document: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de l'ajout du document: {str(e)}"
        }), 500


@documents_bp.route('/documents/upload', methods=['POST'])
def upload_document():
    """
    Upload un fichier et l'ajoute à la base de connaissances.
    
    Form data:
    - file: fichier à uploader
    - metadata: JSON string (optionnel)
    - document_id: string (optionnel)
    
    Returns:
        JSON avec le statut et le nombre de chunks ajoutés
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "Aucun fichier fourni"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "Aucun fichier sélectionné"
            }), 400
        
        if not allowed_file(file.filename):
            from src.rag import DocumentLoader
            supported = DocumentLoader.get_supported_formats()
            return jsonify({
                "status": "error",
                "message": f"Format de fichier non supporté. Formats supportés: {', '.join(supported)}"
            }), 400
        
        # Sauvegarder le fichier temporairement
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Parser les métadonnées si fournies
            metadata = None
            if 'metadata' in request.form:
                try:
                    metadata = json.loads(request.form['metadata'])
                except json.JSONDecodeError:
                    return jsonify({
                        "status": "error",
                        "message": "Le champ 'metadata' doit être un JSON valide"
                    }), 400
            
            document_id = request.form.get('document_id')
            
            # Ajouter le document
            chunks_count = rag_manager.add_document(
                file_path=file_path,
                metadata=metadata,
                document_id=document_id
            )
            
            return jsonify({
                "status": "success",
                "message": "Fichier uploadé et ajouté avec succès",
                "filename": filename,
                "chunks_added": chunks_count
            }), 200
        
        finally:
            # Nettoyer le fichier temporaire
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Impossible de supprimer le fichier temporaire {file_path}: {e}")
    
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du document: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de l'upload: {str(e)}"
        }), 500

