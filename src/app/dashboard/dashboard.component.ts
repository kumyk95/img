import { Component, ElementRef, ViewChild, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SimulationService, SimulationStep } from '../simulation.service';
import { Observable, Subscription } from 'rxjs';

@Component({
    selector: 'app-dashboard',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './dashboard.component.html',
    styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit, OnDestroy {
    @ViewChild('processingCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;

    steps$: Observable<SimulationStep[]>;
    isRunning$: Observable<boolean>;
    isPaused$: Observable<boolean>;

    private subscription?: Subscription;

    // Batch processing state
    filesToProcess: File[] = [];
    currentBatchIndex = -1;
    totalBatchSize = 0;
    batchProgress: number | null = null;
    private isBatchRunning = false;

    constructor(private simulationService: SimulationService) {
        this.steps$ = this.simulationService.steps$;
        this.isRunning$ = this.simulationService.isRunning$;
        this.isPaused$ = this.simulationService.isPaused$;
    }

    ngOnInit() {
        // Listen for simulation completion to proceed to next file in batch
        this.subscription = this.isRunning$.subscribe(running => {
            if (!running && this.isBatchRunning) {
                // Short delay before next image
                setTimeout(() => this.processNextInBatch(), 1000);
            }
        });
    }

    ngOnDestroy() {
        this.subscription?.unsubscribe();
    }

    onFileSelected(event: any) {
        const files = (event.target as HTMLInputElement).files;
        if (files && files.length > 0) {
            this.filesToProcess = Array.from(files);
            this.totalBatchSize = this.filesToProcess.length;
            this.currentBatchIndex = -1;
            this.batchProgress = 0;

            // Clear previous batch results
            this.simulationService.clearBatchResults();

            // Load the first file immediately as "Original" preview
            this.loadPreview(this.filesToProcess[0]);
        }
    }

    private loadPreview(file: File) {
        const reader = new FileReader();
        reader.onload = (e: any) => {
            this.simulationService.updateStepImage(1, e.target.result, file);
        };
        reader.readAsDataURL(file);
    }

    run() {
        if (this.filesToProcess.length > 0) {
            this.isBatchRunning = true;
            if (this.currentBatchIndex === -1) {
                this.processNextInBatch();
            } else {
                this.simulationService.run();
            }
        }
    }

    private processNextInBatch() {
        if (!this.isBatchRunning) return;

        this.currentBatchIndex++;
        if (this.currentBatchIndex < this.totalBatchSize) {
            this.batchProgress = ((this.currentBatchIndex + 1) / this.totalBatchSize) * 100;
            const file = this.filesToProcess[this.currentBatchIndex];

            // Reset steps for new image
            this.simulationService.resetSteps();

            // Set current image index for batch tracking
            this.simulationService.setCurrentImageIndex(this.currentBatchIndex);

            // Load and upload the image, then START simulation
            const reader = new FileReader();
            reader.onload = (e: any) => {
                this.simulationService.updateStepImage(1, e.target.result, file);
                // Wait for upload/session start and then run
                setTimeout(() => this.simulationService.run(), 800);
            };
            reader.readAsDataURL(file);
        } else {
            this.isBatchRunning = false;
            this.batchProgress = 100;
            console.log('Batch processing completed!');
        }
    }

    pause() {
        this.simulationService.pause();
    }

    continue() {
        this.simulationService.continue();
    }

    stop() {
        this.isBatchRunning = false;
        this.batchProgress = null;
        this.simulationService.stop();
    }

    export() {
        this.simulationService.exportData();
    }
}
