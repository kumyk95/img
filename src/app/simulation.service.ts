import { Injectable } from '@angular/core';
import { BehaviorSubject, Subscription } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import * as XLSX from 'xlsx';

export interface SimulationStep {
    id: number;
    name: string;
    description: string;
    imageKey: string;
    imageDataUrl?: string;
    status: 'pending' | 'processing' | 'completed';
    score?: number;
    time?: number;
    recognitionTime?: number;
    fileName?: string;
}

export interface BatchResult {
    imageIndex: number;
    fileName: string;
    sessionId: string;
    timestamp: Date;
    steps: SimulationStep[];
}

@Injectable({
    providedIn: 'root'
})
export class SimulationService {
    private steps: SimulationStep[] = [
        { id: 1, name: 'Original', description: 'Original Photo', imageKey: 'original', status: 'pending' },
        { id: 2, name: 'With Glasses', description: 'Added Glasses + ID', imageKey: 'sunglasses', status: 'pending' },
        { id: 3, name: 'Inpainting', description: 'Recovery (Telea)', imageKey: 'inpainted', status: 'pending' },
        { id: 4, name: 'CNN', description: 'GFPGAN (ResNet)', imageKey: 'ar1', status: 'pending' },
        { id: 5, name: 'GAN', description: 'CodeFormer (Transformer)', imageKey: 'ar2', status: 'pending' },
        { id: 6, name: 'Hybrid', description: 'Ensemble (Best of Both)', imageKey: 'ar3', status: 'pending' }
    ];

    private _steps$ = new BehaviorSubject<SimulationStep[]>(this.steps);
    public steps$ = this._steps$.asObservable();

    private _isRunning$ = new BehaviorSubject<boolean>(false);
    public isRunning$ = this._isRunning$.asObservable();

    private _isPaused$ = new BehaviorSubject<boolean>(false);
    public isPaused$ = this._isPaused$.asObservable();

    private sessionId: string | null = null;
    private currentFile: File | null = null;

    // Batch results storage for Excel export
    private batchResults: BatchResult[] = [];
    private currentImageIndex: number = 0;
    private currentFileName: string = '';

    constructor(private http: HttpClient) { }

    private baseUrl = '/api';

    updateStepImage(id: number, dataUrl: string, file?: File) {
        const step = this.steps.find(s => s.id === id);
        if (step) {
            step.imageDataUrl = dataUrl;
            step.status = 'completed';
            if (id === 1 && file) {
                step.fileName = file.name;
            }
            this._steps$.next([...this.steps]);

            if (id === 1 && file) {
                this.currentFile = file;
                this.currentFileName = file.name;
                this.uploadOriginalImage(file);
            }
        }
    }

    private uploadOriginalImage(file: File) {
        const formData = new FormData();
        formData.append('file', file, file.name);
        this.http.post<any>(`${this.baseUrl}/upload`, formData).subscribe({
            next: (res: any) => {
                this.sessionId = res.session_id;
                console.log('Image uploaded, session:', this.sessionId);
            },
            error: (err: any) => console.error('Upload failed', err)
        });
    }

    run() {
        if (!this.sessionId) {
            // Wait for upload if not yet done
            setTimeout(() => {
                if (this.sessionId) this.run();
                else console.warn('Waiting for upload to complete...');
            }, 500);
            return;
        }

        if (this._isRunning$.value && !this._isPaused$.value) return;

        this._isRunning$.next(true);
        this._isPaused$.next(false);

        // Start from first non-completed step, or Step 2
        const nextStep = this.steps.find(s => s.status === 'pending');
        if (nextStep) {
            this.processStep(nextStep.id);
        } else {
            this._isRunning$.next(false);
        }
    }

    private processStep(stepId: number) {
        if (!this._isRunning$.value || this._isPaused$.value || !this.sessionId) return;

        const index = this.steps.findIndex(s => s.id === stepId);
        if (index === -1) {
            this._isRunning$.next(false);
            return;
        }

        const step = this.steps[index];
        step.status = 'processing';
        this._steps$.next([...this.steps]);

        let endpoint = '';
        if (stepId === 2) endpoint = '/process/glasses';
        else if (stepId === 3) endpoint = '/process/remove-glasses';
        else {
            endpoint = `/process/method/${stepId}`;
        }

        this.http.post<any>(`${this.baseUrl}${endpoint}?session_id=${this.sessionId}`, {}).subscribe({
            next: (res: any) => {
                step.status = 'completed';
                step.imageDataUrl = res.image_url;
                step.score = res.score;
                step.time = res.time_ms;
                step.recognitionTime = res.recognition_time_ms;
                (step as any).retryCount = 0; // Reset retry count on success
                this._steps$.next([...this.steps]);

                setTimeout(() => {
                    if (this._isRunning$.value && !this._isPaused$.value) {
                        const nextStep = this.steps[index + 1];
                        if (nextStep) {
                            this.processStep(nextStep.id);
                        } else {
                            // All steps completed for this image - save to batch results
                            this.saveBatchResult();
                            this._isRunning$.next(false);
                        }
                    }
                }, 800);
            },
            error: (err: any) => {
                console.error(`Step ${stepId} failed:`, err);

                // Retry logic: try up to 3 times
                const retryCount = (step as any).retryCount || 0;
                if (retryCount < 3) {
                    console.log(`Retrying step ${stepId}, attempt ${retryCount + 1}/3`);
                    (step as any).retryCount = retryCount + 1;
                    step.status = 'processing'; // Keep as processing during retry
                    setTimeout(() => {
                        if (this._isRunning$.value && !this._isPaused$.value) {
                            this.processStep(stepId);
                        }
                    }, 2000); // Wait 2 seconds before retry
                } else {
                    // Max retries reached, mark as failed and continue to next step
                    console.error(`Step ${stepId} failed after 3 retries, skipping...`);
                    step.status = 'pending'; // Mark as failed
                    (step as any).retryCount = 0; // Reset retry count

                    // Continue to next step instead of stopping entire batch
                    setTimeout(() => {
                        if (this._isRunning$.value && !this._isPaused$.value) {
                            const nextStep = this.steps[index + 1];
                            if (nextStep) {
                                this.processStep(nextStep.id);
                            } else {
                                // Save partial results
                                this.saveBatchResult();
                                this._isRunning$.next(false);
                            }
                        }
                    }, 800);
                }
            }
        });
    }

    private saveBatchResult() {
        // Save current image results to batch
        const result: BatchResult = {
            imageIndex: this.currentImageIndex,
            fileName: this.currentFileName,
            sessionId: this.sessionId || '',
            timestamp: new Date(),
            steps: JSON.parse(JSON.stringify(this.steps)) // Deep copy
        };
        this.batchResults.push(result);
        console.log(`Saved batch result for image ${this.currentImageIndex + 1}: ${this.currentFileName}`);
    }

    pause() {
        this._isPaused$.next(true);
    }

    continue() {
        this._isPaused$.next(false);
        this._isRunning$.next(true);
        const nextStep = this.steps.find(s => s.status === 'pending');
        if (nextStep) this.processStep(nextStep.id);
    }

    stop() {
        this._isRunning$.next(false);
        this._isPaused$.next(false);
        this.resetSteps();
        this.sessionId = null;
    }

    resetSteps() {
        this.steps.forEach((s, idx) => {
            s.status = 'pending';
            s.score = undefined;
            s.time = undefined;
            s.recognitionTime = undefined;
            (s as any).retryCount = 0; // Reset retry count
            if (idx !== 0) s.imageDataUrl = undefined;
        });
        this._steps$.next([...this.steps]);
    }

    // New method to set current image index (called from dashboard)
    setCurrentImageIndex(index: number) {
        this.currentImageIndex = index;
    }

    // Clear batch results (called when starting new batch)
    clearBatchResults() {
        this.batchResults = [];
        this.currentImageIndex = 0;
    }

    exportData() {
        if (this.batchResults.length === 0) {
            // Fallback to old CSV export for single image
            const csvContent = "data:text/csv;charset=utf-8,"
                + "ID,Name,Status,Score,ProcTime(ms),RecTime(ms)\n"
                + this.steps.map(e => `${e.id},${e.name},${e.status},${e.score?.toFixed(4) || ''},${e.time?.toFixed(0) || ''},${e.recognitionTime?.toFixed(0) || ''}`).join("\n");

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `results_${this.sessionId || 'exp'}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            // Export batch results to Excel
            this.exportBatchToExcel();
        }
    }

    private exportBatchToExcel() {
        // Create Excel workbook
        const wb = XLSX.utils.book_new();

        // Prepare data for Excel
        const excelData: any[] = [];

        // Add header row
        excelData.push([
            'Image #',
            'File Name',
            'Session ID',
            'Timestamp',
            'Original_Status',
            'Glasses_Status',
            'Glasses_Score',
            'Glasses_ProcTime(ms)',
            'Glasses_RecTime(ms)',
            'Inpainting_Status',
            'Inpainting_Score',
            'Inpainting_ProcTime(ms)',
            'Inpainting_RecTime(ms)',
            'CNN_Status',
            'CNN_Score',
            'CNN_ProcTime(ms)',
            'CNN_RecTime(ms)',
            'GAN_Status',
            'GAN_Score',
            'GAN_ProcTime(ms)',
            'GAN_RecTime(ms)',
            'Hybrid_Status',
            'Hybrid_Score',
            'Hybrid_ProcTime(ms)',
            'Hybrid_RecTime(ms)'
        ]);

        // Add data rows
        this.batchResults.forEach(result => {
            const row = [
                result.imageIndex + 1,
                result.fileName,
                result.sessionId,
                result.timestamp.toLocaleString(),
            ];

            // Add data for each step
            result.steps.forEach(step => {
                row.push(step.status);
                if (step.id > 1) { // Skip Original (no score/time)
                    row.push(step.score?.toFixed(4) || '');
                    row.push(step.time?.toFixed(0) || '');
                    row.push(step.recognitionTime?.toFixed(0) || '');
                }
            });

            excelData.push(row);
        });

        // Create worksheet
        const ws = XLSX.utils.aoa_to_sheet(excelData);

        // Set column widths
        ws['!cols'] = [
            { wch: 10 },  // Image #
            { wch: 30 },  // File Name
            { wch: 40 },  // Session ID
            { wch: 20 },  // Timestamp
            ...Array(21).fill({ wch: 15 })  // All other columns
        ];

        // Add worksheet to workbook
        XLSX.utils.book_append_sheet(wb, ws, 'Batch Results');

        // Generate Excel file and download
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        XLSX.writeFile(wb, `batch_results_${timestamp}.xlsx`);

        console.log(`Exported ${this.batchResults.length} images to Excel`);
    }
}
