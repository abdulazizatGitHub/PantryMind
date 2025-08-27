import { useState, useEffect, useRef } from 'react';
import API from '../utils/api';
import Webcam from 'react-webcam';

export default function Pantry(){
    const [items, setItems] = useState([]);
    const [newItem, setNewItem] = useState('');
    const webcamRef = useRef(null);
    
    
    useEffect(()=>{ fetchPantry(); }, []);

    const fetchPantry = async ()=>{
        const res = await API.get('/pantry/list');
        setItems(res.data);
    }

    const addItem = async ()=>{
        if(!newItem) return;
        await API.post('/pantry/upsert', {name:newItem, quantity:1});
        setNewItem('');
        fetchPantry();
    }

    const capture = async ()=>{
        const imageSrc = webcamRef.current.getScreenshot({width:1280});
        const blob = await (await fetch(imageSrc)).blob();
        const form = new FormData();
        form.append('image', blob, 'snap.jpg');
        await API.post('/detect-and-upsert', form, { headers: {'Content-Type':'multipart/form-data'} });
        fetchPantry();
    }

    return (
        <div className="bg-white p-4 rounded shadow">
            <h2 className="font-bold text-lg mb-2">Pantry</h2>
            <div className="mb-2">
                <input value={newItem} onChange={(e)=>setNewItem(e.target.value)} placeholder="Add item..." className="border px-2 py-1 rounded w-full" />
                <button onClick={addItem} className="mt-2 bg-green-600 text-white px-3 py-1 rounded">Add</button>
            </div>
        
            <div className="mb-2">
                <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" className="w-full" />
                <button onClick={capture} className="mt-2 bg-blue-600 text-white px-3 py-1 rounded w-full">Scan with Camera</button>
            </div>
        
            <ul className="space-y-2">
                {items.map(it=> (
                    <li key={it._id} className="flex items-center justify-between p-2 border rounded">
                        <div className="flex items-center gap-3">
                            {it.thumbnail_b64 && <img src={`data:image/jpeg;base64,${it.thumbnail_b64}`} className="w-12 h-12 rounded object-cover" alt="thumb" />}
                            <div>
                                <div className="font-semibold">{it.name}</div>
                                <div className="text-xs text-gray-500">{it.ocr_text}</div>
                            </div>
                        </div>
                        <div className="text-right">
                            <div>{it.expiry_date}</div>
                            {it.status === 'red' && <div className="text-red-600 text-sm">Expiring soon</div>}
                            {it.status === 'amber' && <div className="text-amber-600 text-sm">Use soon</div>}
                            {it.status === 'green' && <div className="text-green-600 text-sm">Fresh</div>}
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    )
}
